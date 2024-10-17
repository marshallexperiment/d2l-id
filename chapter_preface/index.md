# Kata Pengantar

Beberapa tahun yang lalu, tidak ada banyak ilmuwan pembelajaran mendalam yang mengembangkan produk dan layanan cerdas di perusahaan besar maupun startup. Ketika kami memasuki bidang ini, pembelajaran mesin tidak menjadi berita utama di koran-koran harian. Orang tua kami tidak tahu apa itu pembelajaran mesin, apalagi mengapa kami lebih memilihnya daripada karir di bidang kedokteran atau hukum. Pembelajaran mesin adalah disiplin akademis yang bersifat teoretis, dan signifikansi industrinya terbatas pada sejumlah kecil aplikasi dunia nyata, termasuk pengenalan suara dan visi komputer. Selain itu, banyak dari aplikasi ini membutuhkan begitu banyak pengetahuan domain sehingga sering dianggap sebagai area yang sama sekali terpisah, di mana pembelajaran mesin hanyalah salah satu komponen kecil. Pada waktu itu, jaringan saraf—pendahulu dari metode pembelajaran mendalam yang menjadi fokus buku ini—umumnya dianggap sudah ketinggalan zaman.

Namun, hanya dalam beberapa tahun, pembelajaran mendalam telah mengejutkan dunia, mendorong kemajuan pesat di berbagai bidang seperti visi komputer, pemrosesan bahasa alami, pengenalan suara otomatis, pembelajaran penguatan, dan informatika biomedis. Selain itu, keberhasilan pembelajaran mendalam dalam begitu banyak tugas yang bernilai praktis bahkan telah memicu perkembangan dalam pembelajaran mesin teoretis dan statistik. Dengan kemajuan ini, kita sekarang dapat membangun mobil yang mengemudi sendiri dengan otonomi yang lebih besar dari sebelumnya (meskipun tidak sebanyak yang mungkin diklaim beberapa perusahaan), sistem dialog yang memperbaiki kode dengan bertanya pertanyaan klarifikasi, dan agen perangkat lunak yang mengalahkan pemain manusia terbaik di dunia dalam permainan papan seperti Go, sebuah prestasi yang dulu dianggap masih puluhan tahun lagi. Alat-alat ini sudah mulai memberikan pengaruh yang semakin luas di industri dan masyarakat, mengubah cara pembuatan film, diagnosis penyakit, dan memainkan peran yang semakin besar dalam sains dasar—mulai dari astrofisika, hingga pemodelan iklim, hingga prediksi cuaca, dan biomedis.


## Tentang Buku Ini

Buku ini adalah upaya kami untuk membuat pembelajaran mendalam lebih mudah diakses, dengan mengajarkan *konsep*, *konteks*, dan *kode*.

### Satu Medium yang Menggabungkan Kode, Matematika, dan HTML

Agar teknologi komputasi dapat mencapai dampak penuhnya, teknologi tersebut harus dipahami dengan baik, didokumentasikan dengan baik, dan didukung oleh alat-alat yang matang dan terpelihara dengan baik. Ide-ide kunci harus disajikan secara jelas, sehingga waktu yang dibutuhkan untuk membawa praktisi baru bisa dipersingkat. Pustaka yang matang harus mengotomatiskan tugas-tugas umum, dan kode contoh harus memudahkan praktisi untuk memodifikasi, menerapkan, dan memperluas aplikasi umum sesuai dengan kebutuhan mereka.

Sebagai contoh, ambil aplikasi web dinamis. Meskipun banyak perusahaan, seperti Amazon, yang berhasil mengembangkan aplikasi web berbasis database pada tahun 1990-an, potensi teknologi ini untuk membantu wirausahawan kreatif baru disadari secara jauh lebih besar dalam sepuluh tahun terakhir, sebagian berkat pengembangan kerangka kerja yang kuat dan terdokumentasi dengan baik.

Menguji potensi pembelajaran mendalam menghadirkan tantangan unik karena setiap aplikasi tunggal menggabungkan berbagai disiplin ilmu. Menerapkan pembelajaran mendalam memerlukan pemahaman secara bersamaan mengenai (i) motivasi untuk merumuskan masalah dengan cara tertentu; (ii) bentuk matematis dari model yang diberikan; (iii) algoritma optimasi untuk menyesuaikan model dengan data; (iv) prinsip-prinsip statistik yang memberi tahu kita kapan kita bisa mengharapkan model kita untuk menggeneralisasi ke data yang tidak terlihat dan metode praktis untuk memastikan bahwa model tersebut benar-benar dapat menggeneralisasi; serta (v) teknik rekayasa yang diperlukan untuk melatih model secara efisien, menghindari jebakan komputasi numerik, dan memaksimalkan penggunaan perangkat keras yang tersedia. Mengajarkan keterampilan berpikir kritis yang diperlukan untuk merumuskan masalah, matematika untuk menyelesaikannya, dan alat perangkat lunak untuk mengimplementasikan solusi tersebut semuanya dalam satu tempat adalah tantangan besar. Tujuan kami dalam buku ini adalah untuk menyediakan sumber daya yang terpadu guna mempercepat pemahaman para calon praktisi.

Ketika kami memulai proyek buku ini, tidak ada sumber daya yang sekaligus (i) tetap terkini; (ii) mencakup luasnya praktik pembelajaran mesin modern dengan kedalaman teknis yang memadai; dan (iii) menggabungkan penjelasan yang memiliki kualitas seperti buku teks dengan kode yang bersih dan dapat dijalankan, seperti yang diharapkan dari tutorial langsung. Kami menemukan banyak contoh kode yang menggambarkan cara menggunakan kerangka pembelajaran mendalam tertentu (misalnya, cara melakukan komputasi numerik dasar dengan matriks di TensorFlow) atau untuk mengimplementasikan teknik tertentu (misalnya, potongan kode untuk LeNet, AlexNet, ResNet, dll.) yang tersebar di berbagai posting blog dan repositori GitHub. Namun, contoh-contoh ini biasanya hanya berfokus pada *bagaimana* mengimplementasikan pendekatan tertentu, tetapi tidak menyertakan diskusi tentang *mengapa* keputusan algoritmik tertentu dibuat. Meskipun beberapa sumber daya interaktif muncul secara sporadis untuk membahas topik tertentu, misalnya, posting blog yang menarik di situs [Distill](http://distill.pub) atau blog pribadi, mereka hanya mencakup topik tertentu dalam pembelajaran mendalam dan sering kali tidak disertai kode yang relevan. Di sisi lain, meskipun beberapa buku teks pembelajaran mendalam telah muncul—misalnya, :citet:`Goodfellow.Bengio.Courville.2016`, yang menawarkan survei komprehensif tentang dasar-dasar pembelajaran mendalam—sumber-sumber ini tidak menghubungkan deskripsi dengan implementasi konsep dalam kode, sering kali membuat pembaca kebingungan tentang bagaimana mengimplementasikannya. Selain itu, terlalu banyak sumber daya yang disembunyikan di balik dinding pembayaran penyedia kursus komersial.

Kami berangkat untuk membuat sumber daya yang dapat
(i) tersedia secara gratis untuk semua orang;
(ii) menawarkan kedalaman teknis yang cukup
untuk menyediakan titik awal bagi mereka yang ingin menjadi ilmuwan pembelajaran mesin terapan;
(iii) menyertakan kode yang dapat dijalankan, menunjukkan kepada pembaca
*bagaimana* memecahkan masalah secara praktik;
(iv) memungkinkan pembaruan cepat, baik oleh kami
maupun oleh komunitas secara umum;
dan (v) dilengkapi dengan [forum](https://discuss.d2l.ai/c/5)
untuk diskusi interaktif tentang detail teknis dan untuk menjawab pertanyaan.

Tujuan-tujuan ini sering kali bertentangan.
Persamaan, teorema, dan kutipan
paling baik dikelola dan disusun dalam LaTeX.
Kode paling baik dijelaskan dalam Python.
Dan halaman web ditulis secara native dalam HTML dan JavaScript.
Selain itu, kami ingin konten ini
dapat diakses baik sebagai kode yang dapat dijalankan, buku fisik,
sebagai PDF yang dapat diunduh, dan di Internet sebagai situs web.
Tidak ada alur kerja yang tampaknya sesuai dengan tuntutan ini,
jadi kami memutuskan untuk merakit sistem kami sendiri (:numref:`sec_how_to_contribute`).
Kami memilih GitHub untuk berbagi sumber
dan untuk memfasilitasi kontribusi komunitas;
Jupyter notebook untuk menggabungkan kode, persamaan, dan teks;
Sphinx sebagai mesin rendering;
dan Discourse sebagai platform diskusi.
Meskipun sistem kami tidak sempurna,
pilihan-pilihan ini memberikan kompromi
di antara berbagai kepentingan yang bersaing.
Kami percaya bahwa *Dive into Deep Learning*
mungkin merupakan buku pertama yang diterbitkan
dengan menggunakan alur kerja yang terintegrasi seperti ini.

### Belajar dengan Melakukan

Banyak buku teks menyajikan konsep secara berurutan,
mencakup setiap konsep secara mendetail.
Misalnya, buku teks yang sangat baik dari
:citet:`Bishop.2006`
mengajarkan setiap topik dengan sangat mendalam
sehingga untuk sampai ke bab tentang regresi linier
membutuhkan usaha yang tidak sedikit.
Meskipun para ahli menyukai buku ini
karena sifatnya yang sangat mendetail,
bagi pemula sejati, sifat ini membatasi
kegunaannya sebagai teks pengantar.

Dalam buku ini, kami mengajarkan sebagian besar konsep *tepat waktu*.
Dengan kata lain, Anda akan mempelajari konsep-konsep tersebut tepat pada saat
konsep-konsep itu diperlukan untuk mencapai tujuan praktis tertentu.
Meskipun kami meluangkan sedikit waktu di awal untuk mengajarkan
dasar-dasar penting, seperti aljabar linier dan probabilitas,
kami ingin Anda merasakan kepuasan dari melatih model pertama Anda
sebelum khawatir tentang konsep-konsep yang lebih rumit.

Selain dari beberapa notebook awal yang memberikan kursus singkat
tentang latar belakang matematika dasar,
setiap bab berikutnya memperkenalkan sejumlah konsep baru yang wajar
dan menyediakan beberapa contoh yang dapat dijalankan, menggunakan dataset nyata.
Ini menghadirkan tantangan dalam hal organisasi.
Beberapa model secara logis mungkin dikelompokkan bersama dalam satu notebook.
Dan beberapa ide mungkin paling baik diajarkan
dengan menjalankan beberapa model secara berurutan.
Sebaliknya, ada keuntungan besar dalam mengikuti
kebijakan *satu contoh yang dapat dijalankan, satu notebook*:
Ini membuatnya semudah mungkin bagi Anda untuk
memulai proyek penelitian Anda sendiri dengan memanfaatkan kode kami.
Cukup salin sebuah notebook dan mulai memodifikasinya.

Sepanjang buku ini, kami menggabungkan kode yang dapat dijalankan
dengan materi latar belakang yang diperlukan.
Secara umum, kami cenderung membuat alat-alat tersebut tersedia
sebelum menjelaskannya sepenuhnya (seringkali mengisi latar belakang nanti).
Sebagai contoh, kami mungkin menggunakan *stochastic gradient descent*
sebelum menjelaskan mengapa itu berguna
atau memberikan intuisi tentang mengapa itu berhasil.
Hal ini membantu memberikan amunisi yang diperlukan
kepada praktisi untuk memecahkan masalah dengan cepat,
dengan konsekuensi bahwa pembaca harus mempercayai kami
dalam beberapa keputusan kuratorial.

Buku ini mengajarkan konsep pembelajaran mendalam dari awal.
Terkadang, kami menyelami detail-detail mendalam tentang model
yang biasanya disembunyikan dari pengguna
oleh kerangka kerja pembelajaran mendalam modern.
Ini terutama muncul dalam tutorial dasar,
di mana kami ingin Anda memahami segala sesuatu
yang terjadi dalam lapisan atau pengoptimal tertentu.
Dalam kasus ini, kami sering menyajikan
dua versi dari contoh tersebut:
satu di mana kami mengimplementasikan semuanya dari awal,
hanya mengandalkan fungsi mirip NumPy
dan diferensiasi otomatis,
dan contoh yang lebih praktis,
di mana kami menulis kode singkat
menggunakan API tingkat tinggi dari kerangka kerja pembelajaran mendalam.
Setelah menjelaskan bagaimana suatu komponen bekerja,
kami akan mengandalkan API tingkat tinggi di tutorial berikutnya.

### Konten dan Struktur

Buku ini dapat dibagi menjadi tiga bagian utama,
yang membahas tentang dasar-dasar, 
teknik pembelajaran mendalam, 
dan topik-topik lanjutan
yang berfokus pada sistem nyata
dan aplikasi (:numref:`fig_book_org`).

![Struktur Buku.](../img/book-org.svg)
:label:`fig_book_org`

* **Bagian 1: Dasar-dasar dan Pendahuluan**.
:numref:`chap_introduction` adalah
pengantar ke pembelajaran mendalam.
Kemudian, di :numref:`chap_preliminaries`,
kami dengan cepat membawa Anda
pada prasyarat yang dibutuhkan
untuk pembelajaran mendalam secara praktis,
seperti cara menyimpan dan memanipulasi data,
dan bagaimana menerapkan berbagai operasi numerik
berdasarkan konsep dasar dari aljabar linier,
kalkulus, dan probabilitas.
:numref:`chap_regression` dan :numref:`chap_perceptrons`
mencakup konsep dan teknik paling mendasar dalam pembelajaran mendalam,
termasuk regresi dan klasifikasi;
model linier; multilayer perceptrons;
dan overfitting serta regularisasi.

* **Bagian 2: Teknik Pembelajaran Mendalam Modern**.
:numref:`chap_computation` menjelaskan
komponen komputasi kunci
dari sistem pembelajaran mendalam
dan meletakkan dasar
untuk implementasi model yang lebih kompleks
pada bab-bab berikutnya.
Selanjutnya, :numref:`chap_cnn` dan :numref:`chap_modern_cnn`
memperkenalkan convolutional neural networks (CNNs),
alat yang kuat yang menjadi tulang punggung
sebagian besar sistem visi komputer modern.
Demikian pula, :numref:`chap_rnn` dan :numref:`chap_modern_rnn`
memperkenalkan recurrent neural networks (RNNs),
model yang memanfaatkan struktur berurutan (misalnya, temporal)
dalam data dan biasanya digunakan
untuk pemrosesan bahasa alami
dan prediksi deret waktu.
Di :numref:`chap_attention-and-transformers`,
kami menjelaskan kelas model yang relatif baru,
berdasarkan mekanisme *attention*,
yang telah menggantikan RNN sebagai arsitektur dominan
untuk sebagian besar tugas pemrosesan bahasa alami.
Bagian-bagian ini akan membuat Anda memahami
alat paling kuat dan umum
yang banyak digunakan oleh praktisi pembelajaran mendalam.

* **Bagian 3: Skalabilitas, Efisiensi, dan Aplikasi** (tersedia [online](https://d2l.ai)).
Pada Bab 12,
kami membahas beberapa algoritma optimasi umum
yang digunakan untuk melatih model pembelajaran mendalam.
Selanjutnya, pada Bab 13,
kami memeriksa beberapa faktor kunci
yang memengaruhi kinerja komputasi
dari kode pembelajaran mendalam.
Kemudian, pada Bab 14,
kami mengilustrasikan aplikasi-aplikasi utama
pembelajaran mendalam dalam visi komputer.
Terakhir, pada Bab 15 dan Bab 16,
kami mendemonstrasikan cara melatih model representasi bahasa
dan menerapkannya pada tugas-tugas pemrosesan bahasa natural.


### Code
:label:`sec_code`

Sebagian besar bagian dalam buku ini menampilkan kode yang dapat dijalankan.
Kami percaya bahwa beberapa intuisi paling baik dikembangkan
melalui percobaan dan kesalahan,
dengan mengubah sedikit kode dan mengamati hasilnya.
Idealnya, teori matematika yang elegan mungkin bisa memberi tahu kita
secara tepat bagaimana mengubah kode kita untuk mencapai hasil yang diinginkan.
Namun, praktisi pembelajaran mendalam saat ini
sering kali harus melangkah ke wilayah di mana belum ada teori yang kuat sebagai panduan.
Meskipun kami telah berusaha sebaik mungkin, penjelasan formal
mengenai keefektifan berbagai teknik masih kurang, karena beberapa alasan: matematika yang diperlukan untuk mencirikan model-model ini
bisa sangat sulit; penjelasannya mungkin bergantung pada sifat data
yang saat ini masih kurang jelas definisinya;
dan penyelidikan serius tentang topik ini baru-baru ini dimulai secara intensif.
Kami berharap seiring dengan kemajuan teori pembelajaran mendalam,
setiap edisi buku ini di masa mendatang akan memberikan wawasan
yang melampaui apa yang tersedia saat ini.

Untuk menghindari pengulangan yang tidak perlu, kami mengumpulkan
beberapa fungsi dan kelas yang paling sering diimpor dan digunakan
ke dalam paket `d2l`.
Sepanjang buku, kami menandai blok kode
(seperti fungsi, kelas, atau kumpulan pernyataan impor) dengan `#@save`
untuk menunjukkan bahwa mereka akan diakses nanti
melalui paket `d2l`.
Kami menawarkan ikhtisar mendetail
tentang kelas dan fungsi ini di :numref:`sec_d2l`.
Paket `d2l` ringan dan hanya memerlukan
dependensi berikut:


```{.python .input}
#@tab all
#@save
import inspect
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`
Sebagian besar kode dalam buku ini didasarkan pada Apache MXNet,
sebuah kerangka kerja open-source untuk pembelajaran mendalam
yang merupakan pilihan utama AWS (Amazon Web Services),
serta banyak perguruan tinggi dan perusahaan.
Semua kode dalam buku ini telah diuji
di versi terbaru MXNet.
Namun, karena perkembangan pembelajaran mendalam yang sangat cepat,
beberapa kode *dalam edisi cetak*
mungkin tidak berfungsi dengan baik di versi MXNet yang akan datang.
Kami berencana untuk terus memperbarui versi daring.
Jika Anda mengalami masalah,
silakan konsultasikan :ref:`chap_installation`
untuk memperbarui kode dan lingkungan runtime Anda.
Berikut adalah daftar dependensi dalam implementasi MXNet kami.
:end_tab:

:begin_tab:`pytorch`
Sebagian besar kode dalam buku ini didasarkan pada PyTorch,
sebuah kerangka kerja open-source yang populer
dan sangat diterima oleh komunitas penelitian pembelajaran mendalam.
Semua kode dalam buku ini telah diuji
di versi stabil terbaru PyTorch.
Namun, karena perkembangan pembelajaran mendalam yang sangat cepat,
beberapa kode *dalam edisi cetak*
mungkin tidak berfungsi dengan baik di versi PyTorch yang akan datang.
Kami berencana untuk terus memperbarui versi daring.
Jika Anda mengalami masalah,
silakan konsultasikan :ref:`chap_installation`
untuk memperbarui kode dan lingkungan runtime Anda.
Berikut adalah daftar dependensi dalam implementasi PyTorch kami.
:end_tab:

:begin_tab:`tensorflow`
Sebagian besar kode dalam buku ini didasarkan pada TensorFlow,
sebuah kerangka kerja open-source untuk pembelajaran mendalam
yang banyak diadopsi dalam industri
dan populer di kalangan peneliti.
Semua kode dalam buku ini telah diuji
di versi stabil terbaru TensorFlow.
Namun, karena perkembangan pembelajaran mendalam yang sangat cepat,
beberapa kode *dalam edisi cetak*
mungkin tidak berfungsi dengan baik di versi TensorFlow yang akan datang.
Kami berencana untuk terus memperbarui versi daring.
Jika Anda mengalami masalah,
silakan konsultasikan :ref:`chap_installation`
untuk memperbarui kode dan lingkungan runtime Anda.
Berikut adalah daftar dependensi dalam implementasi TensorFlow kami.
:end_tab:

:begin_tab:`jax`
Sebagian besar kode dalam buku ini didasarkan pada Jax,
sebuah kerangka kerja open-source yang memungkinkan transformasi fungsi yang dapat digabungkan,
seperti diferensiasi fungsi Python dan NumPy yang sewenang-wenang,
serta kompilasi JIT, vektorisasi, dan banyak lagi!
Jax semakin populer dalam ruang penelitian pembelajaran mesin
dan memiliki API yang mudah dipelajari mirip dengan NumPy.
Bahkan, JAX berusaha mencapai kesamaan 1:1 dengan NumPy,
sehingga beralih dari kode Anda bisa sesederhana mengubah satu pernyataan impor!
Namun, karena perkembangan pembelajaran mendalam yang sangat cepat,
beberapa kode *dalam edisi cetak*
mungkin tidak berfungsi dengan baik di versi Jax yang akan datang.
Kami berencana untuk terus memperbarui versi daring.
Jika Anda mengalami masalah,
silakan konsultasikan :ref:`chap_installation`
untuk memperbarui kode dan lingkungan runtime Anda.
Berikut adalah daftar dependensi dalam implementasi JAX kami.
:end_tab:


```{.python .input}
#@tab mxnet
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from scipy.spatial import distance_matrix
```

```{.python .input}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab jax
#@save
from dataclasses import field
from functools import partial
import flax
from flax import linen as nn
from flax.training import train_state
import jax
from jax import numpy as jnp
from jax import grad, vmap
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from types import FunctionType
from typing import Any
```

### Audiens Sasaran

Buku ini ditujukan untuk mahasiswa (sarjana atau pascasarjana),
insinyur, dan peneliti yang ingin memahami secara mendalam
teknik praktis pembelajaran mendalam.
Karena kami menjelaskan setiap konsep dari awal,
tidak diperlukan latar belakang sebelumnya dalam pembelajaran mendalam atau pembelajaran mesin.
Untuk menjelaskan metode pembelajaran mendalam secara lengkap
dibutuhkan sedikit matematika dan pemrograman,
namun kami hanya mengasumsikan bahwa Anda memiliki pengetahuan dasar,
termasuk sedikit aljabar linier, kalkulus, probabilitas, dan pemrograman Python.
Jika Anda lupa beberapa hal,
[Appendiks daring](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html) menyediakan ulasan
tentang sebagian besar matematika
yang akan Anda temui dalam buku ini.
Biasanya, kami akan lebih memprioritaskan
intuisi dan ide
dibandingkan ketelitian matematis.
Jika Anda ingin memperdalam dasar-dasar ini
di luar prasyarat yang dibutuhkan untuk memahami buku kami,
kami dengan senang hati merekomendasikan beberapa sumber luar biasa lainnya:
*Linear Analysis* oleh :citet:`Bollobas.1999`
mencakup aljabar linier dan analisis fungsional secara mendalam.
*All of Statistics* :cite:`Wasserman.2013`
memberikan pengantar yang luar biasa tentang statistik.
Buku dan [kursus](https://projects.iq.harvard.edu/stat110/home) Joe Blitzstein tentang probabilitas dan inferensi adalah permata pedagogis.
Dan jika Anda belum pernah menggunakan Python sebelumnya,
Anda mungkin ingin melihat [tutorial Python](http://learnpython.org/) ini.


### Notebook, Website, GitHub, dan Forum

Semua notebook kami dapat diunduh dari [website D2L.ai](https://d2l.ai)
dan dari [GitHub](https://github.com/d2l-ai/d2l-en).
Bersamaan dengan buku ini, kami telah meluncurkan forum diskusi
di [discuss.d2l.ai](https://discuss.d2l.ai/c/5).
Setiap kali Anda memiliki pertanyaan tentang bagian mana pun dari buku ini,
Anda dapat menemukan tautan ke halaman diskusi terkait
di akhir setiap notebook.

## Ucapan Terima Kasih

Kami berutang budi kepada ratusan kontributor baik untuk
draft bahasa Inggris maupun bahasa Tiongkok.
Mereka membantu memperbaiki konten dan memberikan masukan yang berharga.
Buku ini awalnya diimplementasikan dengan MXNet sebagai kerangka kerja utama.
Kami berterima kasih kepada Anirudh Dagar dan Yuan Tang atas adaptasi sebagian besar kode MXNet sebelumnya ke dalam implementasi PyTorch dan TensorFlow, masing-masing.
Sejak Juli 2021, kami telah mendesain ulang dan mengimplementasikan ulang buku ini di PyTorch, MXNet, dan TensorFlow, dengan memilih PyTorch sebagai kerangka kerja utama.
Kami berterima kasih kepada Anirudh Dagar atas adaptasi sebagian besar kode PyTorch yang lebih baru ke dalam implementasi JAX.
Kami berterima kasih kepada Gaosheng Wu, Liujun Hu, Ge Zhang, dan Jiehang Xie dari Baidu atas adaptasi sebagian besar kode PyTorch yang lebih baru ke dalam implementasi PaddlePaddle dalam draft bahasa Tiongkok.
Kami berterima kasih kepada Shuai Zhang karena telah mengintegrasikan gaya LaTeX dari penerbit ke dalam pembuatan PDF.

Di GitHub, kami berterima kasih kepada setiap kontributor dari draft bahasa Inggris ini
karena telah membuatnya lebih baik untuk semua orang.
ID GitHub atau nama mereka adalah (tidak dalam urutan tertentu):
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, Yue Ying, tayfununal,
steinsag, charleybeller, Andrew Lumsdaine, Jiekui Zhang, Deepak Pathak, Florian Donhauser, Tim Gates,
Adriaan Tijsseling, Ron Medina, Gaurav Saha, Murat Semerci, Lei Mao, Levi McClenny, Joshua Broyde,
jake221, jonbally, zyhazwraith, Brian Pulfer, Nick Tomasino, Lefan Zhang, Hongshen Yang, Vinney Cavallo,
yuntai, Yuanxiang Zhu, amarazov, pasricha, Ben Greenawald, Shivam Upadhyay, Quanshangze Du, Biswajit Sahoo,
Parthe Pandit, Ishan Kumar, HomunculusK, Lane Schwartz, varadgunjal, Jason Wiener, Armin Gholampoor,
Shreshtha13, eigen-arnav, Hyeonggyu Kim, EmilyOng, Bálint Mucsányi, Chase DuBois, Juntian Tao,
Wenxiang Xu, Lifu Huang, filevich, quake2005, nils-werner, Yiming Li, Marsel Khisamutdinov,
Francesco "Fuma" Fumagalli, Peilin Sun, Vincent Gurgul, qingfengtommy, Janmey Shukla, Mo Shan,
Kaan Sancak, regob, AlexSauer, Gopalakrishna Ramachandra, Tobias Uelwer, Chao Wang, Tian Cao,
Nicolas Corthorn, akash5474, kxxt, zxydi1992, Jacob Britton, Shuangchi He, zhmou, krahets, Jie-Han Chen,
Atishay Garg, Marcel Flygare, adtygan, Nik Vaessen, bolded, Louis Schlessinger, Balaji Varatharajan,
atgctg, Kaixin Li, Victor Barbaros, Riccardo Musto, Elizabeth Ho, azimjonn, Guilherme Miotto, Alessandro Finamore,
Joji Joseph, Anthony Biel, Zeming Zhao, shjustinbaek, gab-chen, nantekoto, Yutaro Nishiyama, Oren Amsalem,
Tian-MaoMao, Amin Allahyar, Gijs van Tulder, Mikhail Berkov, iamorphen, Matthew Caseres, Andrew Walsh,
pggPL, RohanKarthikeyan, Ryan Choi, and Likun Lei.

Kami berterima kasih kepada Amazon Web Services, terutama Wen-Ming Ye, George Karypis, Swami Sivasubramanian, Peter DeSantis, Adam Selipsky,
dan Andrew Jassy atas dukungan besar mereka dalam penulisan buku ini.
Tanpa waktu, sumber daya, diskusi dengan rekan-rekan, dan dorongan berkelanjutan,
buku ini tidak akan terwujud.
Selama persiapan buku ini untuk penerbitan,
Cambridge University Press memberikan dukungan yang sangat baik.
Kami berterima kasih kepada editor kami, David Tranah
atas bantuan dan profesionalismenya.

## Ringkasan

Pembelajaran mendalam telah merevolusi pengenalan pola,
memperkenalkan teknologi yang kini mendukung berbagai teknologi,
di berbagai bidang seperti visi komputer,
pemrosesan bahasa alami,
dan pengenalan suara otomatis.
Untuk berhasil menerapkan pembelajaran mendalam,
Anda harus memahami cara merumuskan masalah,
matematika dasar pemodelan,
algoritma untuk menyesuaikan model Anda dengan data,
dan teknik rekayasa untuk mengimplementasikan semuanya.
Buku ini menyajikan sumber daya yang komprehensif,
termasuk tulisan, gambar, matematika, dan kode, semuanya di satu tempat.

## Latihan

1. Daftarkan akun di forum diskusi buku ini [discuss.d2l.ai](https://discuss.d2l.ai/).
2. Instal Python di komputer Anda.
3. Ikuti tautan di bagian bawah untuk menuju forum, di mana Anda dapat meminta bantuan, mendiskusikan buku, dan menemukan jawaban atas pertanyaan Anda dengan berinteraksi dengan penulis dan komunitas yang lebih luas.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/186)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17963)
:end_tab:

