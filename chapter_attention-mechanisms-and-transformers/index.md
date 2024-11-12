# Mekanisme Attention dan Transformer
:label:`chap_attention-and-transformers`

Tahun-tahun awal ledakan deep learning terutama didorong oleh hasil yang dihasilkan menggunakan multilayer perceptron, convolutional network, dan recurrent network architectures. Menariknya, arsitektur model yang mendasari banyak terobosan deep learning di tahun 2010-an hampir tidak mengalami banyak perubahan dibandingkan dengan pendahulunya meskipun telah melewati hampir 30 tahun. Meskipun banyak inovasi metodologis baru yang masuk ke dalam toolkit praktisi --- seperti ReLU, residual layers, batch normalization, dropout, dan adaptive learning rate schedules --- inti arsitektur dasar tetap mudah dikenali sebagai implementasi yang diperbesar dari ide-ide klasik.

Terlepas dari ribuan makalah yang mengusulkan ide alternatif, model yang mirip dengan classical convolutional neural networks (:numref:`chap_cnn`) tetap *state-of-the-art* dalam computer vision, dan model yang mirip dengan desain asli LSTM oleh Sepp Hochreiter (:numref:`sec_lstm`) mendominasi sebagian besar aplikasi dalam natural language processing (NLP). Hingga saat itu, kemunculan deep learning yang pesat tampaknya terutama disebabkan oleh perubahan sumber daya komputasi yang tersedia (berkat inovasi dalam komputasi paralel dengan GPU) dan ketersediaan sumber data yang besar (berkat penyimpanan yang murah dan layanan Internet). Meskipun faktor-faktor ini mungkin memang menjadi pendorong utama di balik peningkatan kekuatan teknologi ini, kita juga menyaksikan perubahan besar dalam lanskap arsitektur yang dominan.

Saat ini, model dominan untuk hampir semua tugas NLP berbasis arsitektur Transformer. Ketika menghadapi tugas baru di NLP, pendekatan default yang pertama adalah menggunakan model pretrained berbasis Transformer yang besar (misalnya BERT :cite:`Devlin.Chang.Lee.ea.2018`, ELECTRA :cite:`clark2019electra`, RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`, atau Longformer :cite:`beltagy2020longformer`), menyesuaikan lapisan output yang diperlukan, dan melatih model tersebut pada data yang tersedia untuk tugas selanjutnya. Jika Anda mengikuti berita terbaru beberapa tahun terakhir tentang model bahasa besar OpenAI, maka Anda mengikuti diskusi yang berpusat pada model berbasis Transformer GPT-2 dan GPT-3 :cite:`Radford.Wu.Child.ea.2019,brown2020language`. Sementara itu, Vision Transformer telah muncul sebagai model default untuk berbagai tugas vision, termasuk image recognition, object detection, semantic segmentation, dan superresolution :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,liu2021swin`.

Inti dari model Transformer adalah *attention mechanism*, sebuah inovasi yang awalnya dirancang sebagai peningkatan untuk encoder--decoder RNN yang diterapkan pada aplikasi sequence-to-sequence, seperti machine translation :cite:`Bahdanau.Cho.Bengio.2014`. Dalam model sequence-to-sequence awal untuk machine translation :cite:`Sutskever.Vinyals.Le.2014`, seluruh input dikompres oleh encoder menjadi satu vektor berdimensi tetap yang akan diberikan kepada decoder. Intuisi di balik attention adalah bahwa daripada mengompres input, lebih baik jika decoder dapat mengunjungi kembali input sequence pada setiap langkah. Sebagai tambahan, decoder bisa fokus pada bagian tertentu dari input sequence pada langkah decoding tertentu.

Pada mulanya, ide ini sangat sukses sebagai peningkatan untuk RNN yang sudah mendominasi aplikasi machine translation. Transformer kemudian muncul sebagai model tanpa menggunakan koneksi rekursif, tetapi sepenuhnya bergantung pada mekanisme attention untuk menangkap semua hubungan antar token input dan output. Arsitektur ini sangat efektif, dan pada tahun 2018 Transformer mulai muncul di sebagian besar sistem NLP yang *state-of-the-art*. Dominasi Transformers semakin jelas ketika diterapkan dalam paradigma pretraining ini, dan dengan demikian kebangkitan Transformers bertepatan dengan kebangkitan model pretrained berskala besar, yang sekarang kadang-kadang disebut sebagai *foundation models* :cite:`bommasani2021opportunities`.

Dalam bab ini, kami memperkenalkan model attention, dimulai dengan intuisi paling dasar dan instansiasi ide yang paling sederhana. Kami kemudian bekerja hingga arsitektur Transformer, Vision Transformer, dan lanskap model pretrained berbasis Transformer modern.


```toc
:maxdepth: 2

queries-keys-values
attention-pooling
attention-scoring-functions
bahdanau-attention
multihead-attention
self-attention-and-positional-encoding
transformer
vision-transformer
large-pretraining-transformers
```

