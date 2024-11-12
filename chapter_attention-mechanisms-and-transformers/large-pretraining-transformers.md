# Pretraining Skala Besar dengan Transformers
:label:`sec_large-pretraining-transformers`

Selama ini dalam eksperimen klasifikasi gambar dan terjemahan mesin kita,
model-model dilatih pada dataset dengan contoh input-output *dari awal* untuk melakukan tugas spesifik.
Sebagai contoh, Transformer dilatih dengan pasangan bahasa Inggris-Prancis (:numref:`sec_transformer`) sehingga model ini bisa menerjemahkan teks bahasa Inggris menjadi bahasa Prancis.
Akibatnya, setiap model menjadi *ahli spesifik* yang sensitif bahkan terhadap sedikit perubahan dalam distribusi data (:numref:`sec_environment-and-distribution-shift`).
Untuk model yang lebih umum, atau bahkan *generalist* yang lebih kompeten yang bisa melakukan banyak tugas dengan atau tanpa adaptasi,
*pretraining* model pada data besar semakin banyak dilakukan.

Dengan data yang lebih besar untuk pretraining, arsitektur Transformer bekerja lebih baik dengan peningkatan ukuran model dan komputasi pelatihan, menunjukkan perilaku *skala* yang superior.
Secara khusus, performa model bahasa berbasis Transformer berskala sesuai hukum kekuatan dengan jumlah parameter model, token pelatihan, dan komputasi pelatihan :cite:`kaplan2020scaling`.
Kemampuan skalabilitas Transformer juga dibuktikan oleh peningkatan performa secara signifikan dari vision Transformer yang lebih besar yang dilatih pada data yang lebih besar (dibahas dalam :numref:`sec_vision-transformer`).
Cerita sukses terbaru termasuk Gato, model *generalist* yang bisa bermain Atari, memberi caption pada gambar, mengobrol, dan bertindak sebagai robot :cite:`reed2022generalist`. Gato adalah Transformer tunggal yang memiliki kemampuan skalabilitas baik ketika dilatih pada data multimodal yang beragam, termasuk teks, gambar, torsi sendi, dan penekanan tombol.
Penting untuk dicatat, semua data multimodal tersebut di-serialize menjadi urutan token yang datar, yang dapat diproses seperti token teks (:numref:`sec_transformer`) atau patch gambar (:numref:`sec_vision-transformer`) oleh Transformer.

Sebelum keberhasilan yang memikat dari pretraining Transformer pada data multimodal, Transformer secara ekstensif dilatih dengan banyak teks.
Awalnya diusulkan untuk terjemahan mesin, arsitektur Transformer dalam :numref:`fig_transformer` terdiri dari encoder untuk merepresentasikan urutan input dan decoder untuk menghasilkan urutan target.
Pada dasarnya, Transformer bisa digunakan dalam tiga mode yang berbeda: *encoder-only*, *encoder--decoder*, dan *decoder-only*.
Untuk menyimpulkan bab ini, kita akan mengulas tiga mode tersebut dan menjelaskan skalabilitas dalam pretraining Transformer.

## Encoder-Only

Ketika hanya encoder Transformer yang digunakan, urutan token input dikonversi menjadi representasi dengan jumlah yang sama yang kemudian dapat diproyeksikan menjadi output (misalnya klasifikasi). Encoder Transformer terdiri dari lapisan self-attention, di mana semua token input memperhatikan satu sama lain.
Sebagai contoh, vision Transformer yang digambarkan dalam :numref:`fig_vit` merupakan encoder-only, yang mengonversi urutan patch gambar input menjadi representasi dari token "cls".
Karena representasi ini tergantung pada semua token input, ia kemudian diproyeksikan ke dalam label klasifikasi.
Desain ini terinspirasi oleh Transformer encoder-only yang dilatih pada teks: BERT (Bidirectional Encoder Representations from Transformers) :cite:`Devlin.Chang.Lee.ea.2018`.

### Pretraining BERT

![Kiri: Pretraining BERT dengan masked language modeling. Prediksi token "love" yang disembunyikan tergantung pada semua token input sebelum dan sesudah "love". Kanan: Pola perhatian dalam encoder Transformer. Setiap token pada sumbu vertikal memperhatikan semua token input pada sumbu horizontal.](../img/bert-encoder-only.svg)
:label:`fig_bert-encoder-only`

BERT dilatih pada urutan teks menggunakan *masked language modeling*: teks input dengan token yang secara acak disembunyikan dimasukkan ke dalam encoder Transformer untuk memprediksi token yang disembunyikan.
Seperti yang diilustrasikan dalam :numref:`fig_bert-encoder-only`, urutan teks asli "I", "love", "this", "red", "car" diawali dengan token "cls", dan token "mask" secara acak menggantikan "love"; kemudian cross-entropy loss antara token yang disembunyikan "love" dan prediksinya akan diminimalkan selama pretraining.
Perhatikan bahwa tidak ada batasan dalam pola perhatian dari encoder Transformer (kanan dari :numref:`fig_bert-encoder-only`) sehingga semua token dapat memperhatikan satu sama lain.
Oleh karena itu, prediksi "love" bergantung pada token input sebelum dan sesudahnya dalam urutan tersebut. Inilah mengapa BERT disebut sebagai "bidirectional encoder".
Tanpa perlu pelabelan manual, data teks berskala besar dari buku dan Wikipedia dapat digunakan untuk pretraining BERT.

### Fine-Tuning BERT

BERT yang telah dilatih dapat di-*fine-tune* untuk tugas encoding hilir yang melibatkan teks tunggal atau pasangan teks. Selama fine-tuning, lapisan tambahan dapat ditambahkan pada BERT dengan parameter acak: parameter ini dan parameter yang telah dilatih pada BERT akan *diperbarui* untuk menyesuaikan dengan data pelatihan dari tugas hilir.

![Fine-tuning BERT untuk analisis sentimen.](../img/bert-finetune-classification.svg)
:label:`fig_bert-finetune-classification`

:numref:`fig_bert-finetune-classification` menggambarkan fine-tuning BERT untuk analisis sentimen.
Encoder Transformer adalah BERT yang telah dilatih sebelumnya, yang mengambil urutan teks sebagai input dan memasukkan representasi "cls" (representasi global dari input) ke dalam lapisan fully connected tambahan untuk memprediksi sentimen.
Selama fine-tuning, cross-entropy loss antara prediksi dan label pada data analisis sentimen diminimalkan melalui algoritma berbasis gradien, di mana lapisan tambahan dilatih dari awal sementara parameter BERT yang telah dilatih diperbarui.
BERT melakukan lebih dari analisis sentimen.
Representasi bahasa umum yang dipelajari oleh BERT dengan 350 juta parameter dari 250 miliar token pelatihan meningkatkan performa pada tugas bahasa alami seperti klasifikasi teks tunggal, klasifikasi atau regresi pasangan teks, pelabelan teks, dan menjawab pertanyaan.

Anda mungkin mencatat bahwa tugas hilir ini termasuk pemahaman pasangan teks.
Pretraining BERT memiliki loss lain untuk memprediksi apakah satu kalimat langsung mengikuti yang lain.
Namun, loss ini kemudian ditemukan kurang berguna ketika melatih RoBERTa, varian BERT dengan ukuran yang sama, pada 2000 miliar token :cite:`Liu.Ott.Goyal.ea.2019`.
Derivatif lain dari BERT meningkatkan arsitektur model atau objektif pretraining, seperti ALBERT (memaksakan parameter sharing) :cite:`lan2019albert`, SpanBERT (merepresentasikan dan memprediksi span teks) :cite:`joshi2020spanbert`, DistilBERT (lebih ringan melalui knowledge distillation) :cite:`sanh2019distilbert`, dan ELECTRA (deteksi token yang diganti) :cite:`clark2019electra`.
Selain itu, BERT menginspirasi pretraining Transformer dalam penglihatan komputer, seperti pada vision Transformer :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`, Swin Transformer :cite:`liu2021swin`, dan MAE (masked autoencoders) :cite:`he2022masked`.


## Encoder--Decoder

Karena encoder Transformer mengubah urutan token input menjadi jumlah representasi output yang sama, mode encoder-only tidak dapat menghasilkan urutan dengan panjang yang sembarang seperti dalam terjemahan mesin. Seperti yang awalnya diusulkan untuk terjemahan mesin, arsitektur Transformer dapat dilengkapi dengan decoder yang secara autoregresif memprediksi urutan target dengan panjang sembarang, token demi token, tergantung pada output encoder dan output decoder:
(i) untuk melakukan conditioning pada output encoder, encoder--decoder cross-attention (multi-head attention dari decoder dalam :numref:`fig_transformer`) memungkinkan token target untuk memperhatikan *semua* token input;
(ii) conditioning pada output decoder dicapai dengan pola perhatian yang disebut *causal attention* (nama ini umum dalam literatur tetapi menyesatkan karena hampir tidak ada hubungannya dengan studi kausalitas yang sebenarnya) (masked multi-head attention dari decoder dalam :numref:`fig_transformer`), di mana setiap token target hanya dapat memperhatikan token *masa lalu* dan *sekarang* dalam urutan target.

Untuk melatih Transformer encoder--decoder di luar data terjemahan mesin yang berlabel manusia, BART :cite:`lewis2019bart` dan T5 :cite:`raffel2020exploring` adalah dua Transformer encoder--decoder yang diusulkan secara bersamaan, dilatih pada korpus teks skala besar. Keduanya mencoba merekonstruksi teks asli dalam tujuan pretraining mereka, sementara yang pertama menekankan pada memasukkan noise pada input (misalnya masking, penghapusan, pengacakan, dan rotasi) dan yang terakhir menyoroti unifikasi multitask dengan studi ablation yang komprehensif.

### Pretraining T5

Sebagai contoh Transformer encoder--decoder yang telah dilatih sebelumnya, T5 (Text-to-Text Transfer Transformer) menggabungkan banyak tugas sebagai masalah text-to-text yang sama: untuk setiap tugas, input encoder adalah deskripsi tugas (misalnya, "Summarize", ":") diikuti oleh input tugas (misalnya, urutan token dari sebuah artikel), dan decoder memprediksi output tugas (misalnya, urutan token yang merangkum artikel input). Untuk melakukan tugas sebagai text-to-text, T5 dilatih untuk menghasilkan beberapa teks target bergantung pada teks input.

![Kiri: Pretraining T5 dengan memprediksi span berturut-turut. Kalimat asli adalah "I", "love", "this", "red", "car", di mana "love" diganti dengan token khusus “&lt;X&gt;”, dan span berturut-turut "red", "car" diganti dengan token khusus “&lt;Y&gt;”. Urutan target diakhiri dengan token khusus “&lt;Z&gt;”. Kanan: Pola perhatian dalam Transformer encoder--decoder. Dalam encoder self-attention (persegi bawah), semua token input memperhatikan satu sama lain; Dalam encoder--decoder cross-attention (persegi panjang atas), setiap token target memperhatikan semua token input; Dalam decoder self-attention (segitiga atas), setiap token target hanya memperhatikan token target saat ini dan masa lalu (kausal).](../img/t5-encoder-decoder.svg)
:label:`fig_t5-encoder-decoder`

Untuk mendapatkan input dan output dari teks asli mana pun, T5 dilatih untuk memprediksi span berturut-turut. Secara khusus, token dari teks diganti secara acak oleh token khusus di mana setiap span berturut-turut diganti oleh token khusus yang sama. Pertimbangkan contoh dalam :numref:`fig_t5-encoder-decoder`, di mana teks asli adalah "I", "love", "this", "red", "car". Token "love", "red", "car" secara acak diganti oleh token khusus. Karena "red" dan "car" adalah span berturut-turut, mereka diganti dengan token khusus yang sama. Akibatnya, urutan input menjadi "I", "&lt;X&gt;", "this", "&lt;Y&gt;", dan urutan target menjadi "&lt;X&gt;", "love", "&lt;Y&gt;", "red", "car", "&lt;Z&gt;", di mana "&lt;Z&gt;" adalah token khusus lain yang menandai akhir. Seperti yang ditunjukkan dalam :numref:`fig_t5-encoder-decoder`, decoder memiliki pola causal attention untuk mencegah dirinya memperhatikan token masa depan selama prediksi urutan.

Dalam T5, memprediksi span berturut-turut juga disebut sebagai merekonstruksi teks yang rusak. Dengan tujuan ini, T5 dilatih menggunakan 1000 miliar token dari data C4 (Colossal Clean Crawled Corpus), yang terdiri dari teks bahasa Inggris bersih dari web :cite:`raffel2020exploring`.


### Fine-Tuning T5

Mirip dengan BERT, T5 perlu disesuaikan (*fine-tuned*, memperbarui parameter T5)
dengan data pelatihan yang spesifik untuk tugas tertentu
agar dapat melaksanakan tugas tersebut.
Perbedaan utama dari *fine-tuning* BERT meliputi:
(i) Input T5 mencakup deskripsi tugas;
(ii) T5 dapat menghasilkan urutan dengan panjang sembarang
menggunakan decoder Transformer-nya;
(iii) Tidak diperlukan lapisan tambahan.

![Fine-tuning T5 untuk meringkas teks. Baik deskripsi tugas maupun token artikel dimasukkan ke dalam encoder Transformer untuk memprediksi ringkasan.](../img/t5-finetune-summarization.svg)
:label:`fig_t5-finetune-summarization`

:numref:`fig_t5-finetune-summarization` menjelaskan *fine-tuning* T5
dengan menggunakan peringkasan teks sebagai contoh.
Dalam tugas ini, token deskripsi tugas "Summarize", ":"
diikuti dengan token artikel dimasukkan ke dalam encoder.

Setelah dilakukan *fine-tuning*, T5 dengan 11 miliar parameter (T5-11B)
mencapai hasil terkini (*state-of-the-art*) pada beberapa tolok ukur
pengkodean (misalnya, klasifikasi) dan generasi (misalnya, peringkasan).
Sejak dirilis, T5 telah banyak digunakan dalam penelitian selanjutnya.
Sebagai contoh, *switch Transformers* dirancang berdasarkan T5
untuk mengaktifkan sebagian dari parameter demi efisiensi komputasi yang lebih baik :cite:`fedus2022switch`.
Dalam model teks-ke-gambar yang disebut Imagen,
teks dimasukkan ke encoder T5 yang dibekukan (T5-XXL)
dengan 4,6 miliar parameter :cite:`saharia2022photorealistic`.
Contoh-contoh teks-ke-gambar fotorealistik pada :numref:`fig_imagen`
menunjukkan bahwa encoder T5 saja mungkin dapat
merepresentasikan teks secara efektif bahkan tanpa *fine-tuning*.

![Contoh-contoh teks-ke-gambar oleh model Imagen, di mana encoder teks berasal dari T5 (gambar diambil dari :citet:`saharia2022photorealistic`).](../img/imagen.png)
:width:`700px`
:label:`fig_imagen`


## Decoder-Only

Kita telah meninjau Transformer encoder-only dan encoder--decoder.
Sebagai alternatif, Transformer *decoder-only* menghapus seluruh encoder
dan sublayer decoder dengan *encoder--decoder cross-attention*
dari arsitektur encoder--decoder asli yang digambarkan dalam :numref:`fig_transformer`.
Saat ini, Transformer *decoder-only* telah menjadi arsitektur *de facto*
dalam pemodelan bahasa skala besar (:numref:`sec_language-model`),
yang memanfaatkan korpus teks tidak berlabel dalam jumlah besar di dunia
melalui pembelajaran *self-supervised*.

### GPT dan GPT-2

Dengan menggunakan pemodelan bahasa sebagai tujuan pelatihan,
model GPT (*generative pre-training*)
memilih decoder Transformer sebagai dasarnya :cite:`Radford.Narasimhan.Salimans.ea.2018`.

![Kiri: Pretraining GPT dengan pemodelan bahasa. Urutan target adalah urutan input yang digeser satu token. Baik "&lt;bos&gt;" maupun "&lt;eos&gt;" adalah token khusus yang menandai awal dan akhir dari urutan, masing-masing. Kanan: Pola perhatian dalam decoder Transformer. Setiap token pada sumbu vertikal hanya memperhatikan token masa lalunya pada sumbu horizontal (kausal).](../img/gpt-decoder-only.svg)
:label:`fig_gpt-decoder-only`

Mengikuti pelatihan model bahasa autoregresif
sebagaimana dijelaskan dalam :numref:`subsec_partitioning-seqs`,
:numref:`fig_gpt-decoder-only` mengilustrasikan
*pretraining* GPT dengan encoder Transformer,
di mana urutan target adalah urutan input yang digeser satu token.
Perhatikan bahwa pola perhatian dalam decoder Transformer
memaksa setiap token hanya dapat memperhatikan token masa lalunya
(token masa depan tidak dapat diperhatikan karena belum dipilih).

GPT memiliki 100 juta parameter dan perlu
dilakukan *fine-tuning* untuk setiap tugas hilir.
Satu tahun kemudian, model bahasa Transformer-decoder yang jauh lebih besar,
GPT-2, diperkenalkan :cite:`Radford.Wu.Child.ea.2019`.
Dibandingkan dengan decoder Transformer asli dalam GPT, *pre-normalization*
(dibahas dalam :numref:`subsec_vit-encoder`)
dan peningkatan inisialisasi serta *weight-scaling* diadopsi di GPT-2.
Dilatih dengan 40 GB teks, GPT-2 dengan 1,5 miliar parameter
mencapai hasil terkini (*state-of-the-art*) pada tolok ukur pemodelan bahasa
dan hasil yang menjanjikan pada beberapa tugas lain
*tanpa memperbarui parameter atau arsitektur*.



### GPT-3 dan Lebih Lanjut

GPT-2 menunjukkan potensi penggunaan model bahasa yang sama
untuk beberapa tugas tanpa memperbarui model.
Ini lebih efisien secara komputasi daripada *fine-tuning*,
yang memerlukan pembaruan model melalui perhitungan gradien.

![Pembelajaran *zero-shot*, *one-shot*, *few-shot* dalam konteks model bahasa (decoder Transformer). Tidak diperlukan pembaruan parameter.](../img/gpt-3-xshot.svg)
:label:`fig_gpt-3-xshot`

Sebelum menjelaskan penggunaan model bahasa yang lebih efisien secara komputasi
tanpa pembaruan parameter,
ingat :numref:`sec_rnn-scratch` bahwa model bahasa dapat dilatih
untuk menghasilkan urutan teks yang bergantung pada urutan teks *prefix*.
Dengan demikian, model bahasa yang telah dilatih dapat menghasilkan keluaran tugas
sebagai sebuah urutan *tanpa pembaruan parameter*,
bergantung pada urutan masukan yang berisi deskripsi tugas,
contoh input-output spesifik tugas, dan *prompt* (masukan tugas).
Paradigma pembelajaran ini disebut *in-context learning* :cite:`brown2020language`,
yang dapat dikategorikan lebih lanjut menjadi
*zero-shot*, *one-shot*, dan *few-shot*,
ketika tidak ada, satu, dan beberapa contoh input-output spesifik tugas (:numref:`fig_gpt-3-xshot`).

![Performa agregat GPT-3 untuk semua 42 tolok ukur berbasis akurasi (caption diadaptasi dan gambar diambil dari :citet:`brown2020language`).](../img/gpt3-xshot-scaling.png)
:width:`400px`
:label:`fig_gpt3-xshot-scaling`

Ketiga pengaturan ini diuji pada GPT-3 :cite:`brown2020language`,
yang versi terbesarnya menggunakan data dan ukuran model
sekitar dua kali lipat lebih besar daripada GPT-2.
GPT-3 menggunakan arsitektur decoder Transformer yang sama
dengan pendahulunya, GPT-2,
kecuali bahwa pola perhatian
(di sebelah kanan dalam :numref:`fig_gpt-decoder-only`)
lebih jarang pada lapisan bergantian.
Dilatih dengan 300 miliar token,
GPT-3 menunjukkan kinerja yang lebih baik dengan ukuran model yang lebih besar,
di mana performa *few-shot* meningkat paling cepat (:numref:`fig_gpt3-xshot-scaling`).

Model GPT-4 selanjutnya tidak sepenuhnya mengungkapkan detail teknis dalam laporannya :cite:`openai2023gpt4`.
Berbeda dengan pendahulunya, GPT-4
adalah model berskala besar, multimodal yang
dapat menerima input dalam bentuk teks dan gambar
serta menghasilkan keluaran berupa teks.


## Skalabilitas

:numref:`fig_gpt3-xshot-scaling` secara empiris menunjukkan skalabilitas
dari Transformer dalam model bahasa GPT-3.
Untuk pemodelan bahasa, studi empiris yang lebih komprehensif
tentang skalabilitas Transformer telah mengarahkan peneliti untuk melihat
potensi dalam melatih Transformer yang lebih besar dengan lebih banyak data dan komputasi :cite:`kaplan2020scaling`.

![Kinerja model bahasa Transformer meningkat dengan mulus saat kita meningkatkan ukuran model, ukuran dataset, dan jumlah komputasi yang digunakan untuk pelatihan. Untuk performa optimal, ketiga faktor ini harus ditingkatkan secara bersamaan. Kinerja empiris memiliki hubungan hukum pangkat dengan setiap faktor ketika tidak dibatasi oleh dua faktor lainnya (caption diadaptasi dan gambar diambil dari :citet:`kaplan2020scaling`).](../img/scaling-power-law.png)
:width:`700px`
:label:`fig_scaling-power-law3`

Seperti yang ditunjukkan dalam :numref:`fig_scaling-power-law3`,
*power-law scaling* dapat diamati dalam kinerja
terhadap ukuran model (jumlah parameter, tidak termasuk lapisan embedding),
ukuran dataset (jumlah token pelatihan),
dan jumlah komputasi untuk pelatihan (PetaFLOP/s-hari, tidak termasuk lapisan embedding).
Secara umum, peningkatan ketiga faktor ini secara bersamaan mengarah pada kinerja yang lebih baik.
Namun, *bagaimana* meningkatkan ketiganya secara bersamaan
masih menjadi bahan perdebatan :cite:`hoffmann2022training`.

![Pelatihan model bahasa Transformer (gambar diambil dari :citet:`kaplan2020scaling`).](../img/scaling-sample-conv.png)
:width:`700px`
:label:`fig_scaling-sample-conv`

Selain peningkatan kinerja, model besar juga memiliki efisiensi sampel yang lebih baik dibandingkan dengan model kecil. :numref:`fig_scaling-sample-conv` menunjukkan bahwa model besar membutuhkan lebih sedikit sampel pelatihan (token yang diproses) untuk mencapai level yang sama seperti yang dicapai oleh model kecil, dan kinerja meningkat dengan mulus sesuai dengan jumlah komputasi.

![Kinerja GPT-3 (loss validasi cross-entropy) mengikuti tren power-law dengan jumlah komputasi yang digunakan untuk pelatihan. Perilaku power-law yang diamati dalam :citet:`kaplan2020scaling` berlanjut hingga dua kali lipat lebih besar dengan hanya sedikit deviasi dari kurva yang diprediksi. Parameter embedding tidak termasuk dalam komputasi dan jumlah parameter (caption diadaptasi dan gambar diambil dari :citet:`brown2020language`).](../img/scaling-gpt3.png)
:width:`250px`
:label:`fig_scaling-gpt3`

Perilaku skalabilitas empiris dalam :citet:`kaplan2020scaling` telah diuji dalam model Transformer besar berikutnya. Misalnya, GPT-3 mendukung hipotesis ini dengan peningkatan dua kali lipat lebih besar lagi seperti ditunjukkan dalam :numref:`fig_scaling-gpt3`.



## Model Bahasa Skala Besar

Skalabilitas Transformer dalam seri GPT telah menginspirasi model bahasa besar berikutnya.
Decoder Transformer GPT-2 digunakan untuk melatih Megatron-Turing NLG dengan 530 miliar parameter :cite:`smith2022using` menggunakan 270 miliar token pelatihan. Mengikuti desain GPT-2, Gopher dengan 280 miliar parameter :cite:`rae2021scaling` yang dilatih sebelumnya dengan 300 miliar token, menunjukkan kinerja yang kompetitif dalam berbagai tugas. 
Mewarisi arsitektur yang sama dan menggunakan anggaran komputasi yang sama dengan Gopher, Chinchilla :cite:`hoffmann2022training` adalah model yang jauh lebih kecil (70 miliar parameter) yang dilatih lebih lama (1,4 triliun token pelatihan), melampaui Gopher pada banyak tugas dengan lebih menekankan pada jumlah token daripada jumlah parameter.
Untuk melanjutkan garis skalabilitas pemodelan bahasa,
PaLM (Pathway Language Model) :cite:`chowdhery2022palm`, sebuah decoder Transformer dengan 540 miliar parameter dan desain yang dimodifikasi, dilatih dengan 780 miliar token, mengungguli kinerja manusia rata-rata pada benchmark BIG-Bench :cite:`srivastava2022beyond`. Versi berikutnya, PaLM 2 :cite:`anil2023palm`, menyesuaikan skala data dan model secara kasar 1:1 serta meningkatkan kemampuan multibahasa dan penalaran.
Model bahasa besar lainnya, seperti Minerva :cite:`lewkowycz2022solving` yang melatih model generalis lebih lanjut (PaLM) dan Galactica :cite:`taylor2022galactica` yang tidak dilatih pada korpus umum, telah menunjukkan kemampuan penalaran kuantitatif dan ilmiah yang menjanjikan.

Rilis model open-source, seperti OPT (Open Pretrained Transformers) :cite:`zhang2022opt`, BLOOM :cite:`scao2022bloom`, dan FALCON :cite:`penedo2023refinedweb`,
telah mendemokratisasi penelitian dan penggunaan model bahasa besar.
Dengan fokus pada efisiensi komputasi selama inferensi,
Llama 1 yang open-source :cite:`touvron2023llama` mengungguli model yang jauh lebih besar dengan melatih lebih banyak token daripada yang biasanya digunakan. Llama 2 yang diperbarui :cite:`touvron2023llama2` lebih lanjut meningkatkan korpus prapelatihan sebesar 40%, menghasilkan model produk yang dapat menyaingi kinerja model close-source yang kompetitif.

:citet:`wei2022emergent` membahas kemampuan emergen dari model bahasa besar yang muncul pada model-model yang lebih besar, tetapi tidak pada model yang lebih kecil.
Namun, hanya memperbesar ukuran model tidak secara otomatis membuat model mengikuti instruksi manusia dengan lebih baik.
:citet:`wei2021finetuned,sanh2021multitask` menemukan bahwa fine-tuning model bahasa besar pada beragam dataset yang dijelaskan melalui *instruksi* dapat meningkatkan kinerja zero-shot pada tugas-tugas yang tidak terlihat sebelumnya.
Menggunakan *reinforcement learning dari umpan balik manusia*,
:citet:`ouyang2022training` melakukan fine-tuning pada GPT-3
untuk mengikuti berbagai instruksi.
Berdasarkan hasil dari InstructGPT, yang menyelaraskan model bahasa dengan niat manusia melalui fine-tuning :cite:`ouyang2022training`,
[ChatGPT](https://chat.openai.com/)
dapat menghasilkan respons seperti manusia (misalnya, debugging kode dan penulisan kreatif)
berdasarkan percakapan dengan manusia dan dapat melakukan banyak tugas pemrosesan bahasa alami secara zero-shot :cite:`qin2023chatgpt`.
:citet:`bai2022constitutional` menggantikan input manusia (misalnya, data berlabel manusia) dengan output model untuk sebagian mengotomatisasi proses tuning instruksi, yang juga dikenal sebagai *reinforcement learning dari umpan balik AI*.

Model bahasa besar menawarkan prospek menarik
untuk merumuskan input teks yang dapat mendorong model untuk melakukan tugas-tugas yang diinginkan melalui pembelajaran *in-context*, yang juga dikenal sebagai *prompting*.
Menariknya,
*chain-of-thought prompting* :cite:`wei2022chain`,
metode pembelajaran in-context
dengan contoh-contoh "pertanyaan, langkah penalaran antara, jawaban" few-shot,
mengungkapkan kemampuan penalaran kompleks dari model bahasa besar
untuk menyelesaikan tugas penalaran matematika, akal sehat, dan simbolik.
Menyampling berbagai jalur penalaran :cite:`wang2023self`, mendiversifikasi demonstrasi few-shot :cite:`zhang2023automatic`,
dan mereduksi masalah kompleks menjadi sub-masalah :cite:`zhou2023least`
dapat meningkatkan akurasi penalaran. Bahkan, dengan prompt sederhana seperti "Mari berpikir selangkah demi selangkah" sebelum setiap jawaban,
model bahasa besar dapat melakukan penalaran *chain-of-thought zero-shot* dengan akurasi yang cukup baik :cite:`kojima2022large`.
Bahkan untuk input multimodal yang terdiri dari teks dan gambar,
model bahasa dapat melakukan penalaran chain-of-thought multimodal dengan akurasi lebih tinggi dibandingkan hanya menggunakan input teks :cite:`zhang2023multicot`.




## Ringkasan dan Diskusi

Transformer telah diprapelatihan sebagai encoder-only (misalnya, BERT), encoder--decoder (misalnya, T5), dan decoder-only (misalnya, seri GPT). Model yang telah diprapelatihan dapat diadaptasi untuk melakukan berbagai tugas baik dengan pembaruan model (misalnya, fine-tuning) atau tanpa pembaruan model (misalnya, few-shot). Skalabilitas Transformer menunjukkan bahwa kinerja yang lebih baik mendapat manfaat dari model yang lebih besar, lebih banyak data pelatihan, dan lebih banyak komputasi pelatihan. Karena Transformer pertama kali dirancang dan diprapelatihan untuk data teks, bagian ini sedikit condong ke arah pemrosesan bahasa alami. Namun, model-model yang dibahas di atas sering ditemukan dalam model terbaru yang mencakup banyak modalitas. Misalnya,
(i) Chinchilla :cite:`hoffmann2022training` kemudian diperluas menjadi Flamingo :cite:`alayrac2022flamingo`, sebuah model bahasa visual untuk few-shot learning;
(ii) GPT-2 :cite:`Radford.Wu.Child.ea.2019` dan vision Transformer digunakan untuk menyandi teks dan gambar di CLIP (Contrastive Language-Image Pre-training) :cite:`radford2021learning`, di mana embedding gambar dan teks digunakan dalam sistem teks-ke-gambar DALL-E 2 :cite:`ramesh2022hierarchical`. Meskipun belum ada studi sistematis tentang skalabilitas Transformer dalam prapelatihan multimodal, model teks-ke-gambar berbasis Transformer penuh bernama Parti :cite:`yu2022scaling` menunjukkan potensi skalabilitas di berbagai modalitas:
Parti yang lebih besar lebih mampu menghasilkan gambar berkualitas tinggi dan memahami teks yang kaya konten (:numref:`fig_parti`).


![Contoh gambar yang dihasilkan dari teks yang sama oleh model Parti dengan ukuran yang meningkat (350M, 750M, 3B, 20B) (contoh diambil dari :citet:`yu2022scaling`).](../img/parti.png)
:width:`700px`
:label:`fig_parti`



## Latihan

1. Apakah mungkin melakukan fine-tuning T5 menggunakan minibatch yang terdiri dari tugas yang berbeda? Mengapa atau mengapa tidak? Bagaimana dengan GPT-2?
2. Mengingat model bahasa yang kuat, aplikasi apa yang bisa Anda pikirkan?
3. Misalkan Anda diminta melakukan fine-tuning model bahasa untuk melakukan klasifikasi teks dengan menambahkan lapisan tambahan. Di mana Anda akan menambahkannya? Mengapa?
4. Pertimbangkan masalah sequence-to-sequence (misalnya, terjemahan mesin) di mana urutan input selalu tersedia sepanjang prediksi urutan target. Apa saja keterbatasan pemodelan dengan Transformer *decoder-only*? Mengapa?


[Diskusi](https://discuss.d2l.ai/t/9232)
