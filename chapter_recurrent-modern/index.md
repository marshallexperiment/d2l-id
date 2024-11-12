# RNN Modern

Bab sebelumnya memperkenalkan ide-ide kunci 
di balik jaringan saraf berulang (recurrent neural networks/RNNs). 
Namun, seperti halnya dengan jaringan saraf konvolusional (CNNs),
ada banyak inovasi dalam arsitektur RNN,
yang akhirnya menghasilkan beberapa desain kompleks 
yang terbukti berhasil dalam praktiknya. 
Secara khusus, desain paling populer 
menampilkan mekanisme untuk mengurangi 
ketidakstabilan numerik yang terkenal pada RNN,
seperti yang ditandai oleh gradient vanishing dan exploding gradient.
Ingat bahwa dalam :numref:`chap_rnn`, kita menangani 
exploding gradient dengan menggunakan teknik 
heuristik clipping gradient yang sederhana. 
Meskipun teknik ini efektif, 
teknik ini belum menyelesaikan masalah gradient vanishing. 

Di bab ini, kami memperkenalkan ide-ide kunci di balik 
arsitektur RNN paling sukses untuk data urutan,
yang berasal dari dua makalah. 
Makalah pertama, *Long Short-Term Memory* :cite:`Hochreiter.Schmidhuber.1997`,
memperkenalkan *memory cell* (sel memori), 
sebuah unit komputasi yang menggantikan 
node tradisional di lapisan tersembunyi jaringan.
Dengan memory cell ini, jaringan mampu 
mengatasi kesulitan dalam pelatihan 
yang dihadapi oleh RNN sebelumnya.
Secara intuitif, memory cell menghindari 
masalah gradient vanishing dengan menjaga nilai 
dalam status internal setiap memory cell
yang mengalir melalui edge berulang dengan bobot 1 
di banyak langkah waktu berturut-turut. 
Sekelompok gerbang (gates) multiplikatif membantu jaringan
menentukan tidak hanya input yang boleh masuk 
ke status memori, tetapi juga kapan isi status memori 
harus memengaruhi output model. 

Makalah kedua, *Bidirectional Recurrent Neural Networks* :cite:`Schuster.Paliwal.1997`,
memperkenalkan arsitektur di mana informasi 
dari masa depan (langkah waktu berikutnya) 
dan masa lalu (langkah waktu sebelumnya)
digunakan untuk menentukan output 
pada titik mana pun dalam urutan.
Ini berbeda dengan jaringan sebelumnya, 
di mana hanya input dari masa lalu yang dapat memengaruhi output.
Bidirectional RNN telah menjadi arsitektur utama 
untuk tugas pelabelan urutan dalam pemrosesan bahasa alami,
di antara banyak tugas lainnya. 
Untungnya, kedua inovasi ini tidak saling eksklusif 
dan telah berhasil digabungkan untuk klasifikasi fonem
:cite:`Graves.Schmidhuber.2005` dan pengenalan tulisan tangan :cite:`graves2008novel`.

Bagian pertama dari bab ini akan menjelaskan arsitektur LSTM,
versi yang lebih ringan bernama Gated Recurrent Unit (GRU),
ide-ide utama di balik Bidirectional RNN 
dan penjelasan singkat tentang cara lapisan RNN 
digabungkan untuk membentuk Deep RNN. 
Selanjutnya, kita akan mengeksplorasi aplikasi RNN
dalam tugas sequence-to-sequence, 
memperkenalkan penerjemahan mesin (machine translation)
bersama dengan ide-ide kunci seperti arsitektur *encoder--decoder* dan *beam search*.


```toc
:maxdepth: 2

lstm
gru
deep-rnn
bi-rnn
machine-translation-and-dataset
encoder-decoder
seq2seq
beam-search
```

