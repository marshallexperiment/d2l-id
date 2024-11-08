# Multilayer Perceptrons
:label:`chap_perceptrons`

Dalam bab ini, kita akan memperkenalkan jaringan pertama Anda yang benar-benar *dalam*.
Jaringan dalam yang paling sederhana disebut *multilayer perceptrons*,
dan jaringan ini terdiri dari beberapa lapisan neuron
yang masing-masing terhubung penuh dengan lapisan di bawahnya
(dari mana mereka menerima input)
dan lapisan di atasnya (yang kemudian mereka pengaruhi).
Meskipun diferensiasi otomatis
secara signifikan menyederhanakan implementasi algoritma deep learning,
kita akan mendalami cara perhitungan gradien ini
dalam jaringan yang dalam.
Kemudian kita akan
siap untuk
membahas masalah yang berkaitan dengan stabilitas numerik dan inisialisasi parameter
yang menjadi kunci dalam melatih jaringan dalam dengan sukses.
Ketika kita melatih model dengan kapasitas tinggi seperti ini, kita berisiko mengalami overfitting. Oleh karena itu, kita akan
meninjau kembali regularisasi dan generalisasi
untuk jaringan dalam.
Sepanjang pembahasan ini, kami bertujuan
memberikan pemahaman yang kuat, tidak hanya tentang konsep-konsep ini tetapi juga tentang praktik penggunaan jaringan dalam.
Di akhir bab ini, kita akan menerapkan apa yang telah kita pelajari sejauh ini pada kasus nyata: prediksi harga rumah.
Masalah terkait kinerja komputasi, skalabilitas, dan efisiensi
model kita akan dibahas pada bab-bab selanjutnya.


```toc
:maxdepth: 2

mlp
mlp-implementation
backprop
numerical-stability-and-init
generalization-deep
dropout
kaggle-house-price
```

