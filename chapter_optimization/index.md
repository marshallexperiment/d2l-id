# Algoritma Optimasi
:label:`chap_optimization`

Jika Anda membaca buku ini secara berurutan hingga titik ini, Anda telah menggunakan sejumlah algoritma optimasi untuk melatih model deep learning.
Mereka adalah alat yang memungkinkan kita untuk terus memperbarui parameter model dan meminimalkan nilai fungsi loss, yang dievaluasi pada set pelatihan. Memang, siapa pun yang puas dengan menganggap optimasi sebagai perangkat kotak hitam untuk meminimalkan fungsi objektif dalam pengaturan sederhana mungkin akan cukup dengan mengetahui bahwa ada serangkaian prosedur (dengan nama seperti "SGD" dan "Adam").

Namun, untuk mencapai hasil yang baik, diperlukan pengetahuan yang lebih dalam.
Algoritma optimasi sangat penting untuk deep learning.
Di satu sisi, melatih model deep learning yang kompleks dapat memakan waktu berjam-jam, berhari-hari, atau bahkan berminggu-minggu.
Kinerja algoritma optimasi secara langsung memengaruhi efisiensi pelatihan model.
Di sisi lain, memahami prinsip-prinsip dari berbagai algoritma optimasi dan peran hyperparameter mereka
akan memungkinkan kita untuk menyesuaikan hyperparameter secara terarah untuk meningkatkan kinerja model deep learning.

Dalam bab ini, kita akan menjelajahi algoritma optimasi deep learning yang umum secara mendalam.
Hampir semua masalah optimasi yang muncul dalam deep learning adalah *nonconvex*.
Meskipun demikian, desain dan analisis algoritma dalam konteks masalah *convex* terbukti sangat instruktif.
Oleh karena itu, bab ini mencakup primer tentang optimasi convex dan bukti untuk algoritma stochastic gradient descent yang sangat sederhana pada fungsi objektif convex.


```toc
:maxdepth: 2

optimization-intro
convexity
gd
sgd
minibatch-sgd
momentum
adagrad
rmsprop
adadelta
adam
lr-scheduler
```

