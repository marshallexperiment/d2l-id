# Panduan Pembuat (Builders Guide)
:label:`chap_computation`

Selain dataset besar dan perangkat keras yang kuat,
alat perangkat lunak yang hebat telah memainkan peran tak tergantikan
dalam kemajuan pesat deep learning.
Dimulai dengan pustaka Theano yang inovatif dirilis pada tahun 2007,
alat sumber terbuka yang fleksibel telah memungkinkan para peneliti
untuk membuat prototipe model dengan cepat, menghindari pekerjaan berulang
saat mendaur ulang komponen standar
sambil tetap memungkinkan modifikasi tingkat rendah.
Seiring waktu, pustaka deep learning telah berkembang
untuk menawarkan abstraksi yang semakin tinggi.
Seperti halnya perancang semikonduktor yang beralih dari menentukan transistor
ke sirkuit logika hingga menulis kode,
peneliti jaringan neural juga beralih dari memikirkan
perilaku neuron buatan individual
ke merancang jaringan dalam bentuk keseluruhan lapisan,
dan sekarang sering kali mendesain arsitektur dengan *blok* yang lebih besar dalam pikiran.

Sejauh ini, kami telah memperkenalkan beberapa konsep dasar machine learning,
hingga membuat model deep learning yang sepenuhnya berfungsi.
Pada bab sebelumnya,
kami mengimplementasikan setiap komponen dari sebuah MLP dari awal
dan bahkan menunjukkan cara memanfaatkan API tingkat tinggi
untuk menghasilkan model yang sama dengan mudah.
Untuk mencapai hal itu dengan cepat, kami *menggunakan* pustaka yang tersedia,
tetapi melewatkan detail lebih lanjut tentang *cara kerjanya*.
Di bab ini, kami akan membuka tabir,
membahas lebih dalam komponen utama dari komputasi deep learning,
yaitu konstruksi model, akses dan inisialisasi parameter,
desain lapisan dan blok kustom, membaca dan menulis model ke disk,
dan memanfaatkan GPU untuk mencapai peningkatan kecepatan yang dramatis.
Pemahaman ini akan mengubah Anda dari *pengguna akhir* menjadi *pengguna tingkat lanjut*,
memberikan alat yang diperlukan untuk memanfaatkan
pustaka deep learning yang matang sambil tetap fleksibel
untuk mengimplementasikan model yang lebih kompleks, termasuk yang Anda ciptakan sendiri!
Meskipun bab ini tidak memperkenalkan model atau dataset baru,
bab lanjutan yang mengikutinya sangat bergantung pada teknik-teknik ini.


```toc
:maxdepth: 2

model-construction
parameters
init-param
lazy-init
custom-layer
read-write
use-gpu
```

