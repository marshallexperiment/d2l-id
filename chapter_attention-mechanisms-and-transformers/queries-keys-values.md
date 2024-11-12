```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

# Queries, Keys, dan Values
:label:`sec_queries-keys-values`

Sejauh ini, semua jaringan yang telah kita bahas sangat bergantung pada input dengan ukuran yang telah didefinisikan dengan baik. Misalnya, gambar di ImageNet berukuran $224 \times 224$ piksel, dan CNN khusus disetel untuk ukuran ini. Bahkan dalam natural language processing, ukuran input untuk RNN telah didefinisikan dengan baik dan tetap. Ukuran variabel diatasi dengan memproses satu token pada satu waktu secara berurutan, atau dengan kernel konvolusi yang dirancang khusus :cite:`Kalchbrenner.Grefenstette.Blunsom.2014`. Pendekatan ini dapat menimbulkan masalah signifikan ketika input benar-benar bervariasi dalam ukuran dan konten informasi, seperti dalam :numref:`sec_seq2seq` dalam transformasi teks :cite:`Sutskever.Vinyals.Le.2014`. Terutama untuk urutan panjang, akan menjadi sangat sulit untuk melacak segala sesuatu yang telah dihasilkan atau bahkan dilihat oleh jaringan. Bahkan heuristik pelacakan eksplisit seperti yang diusulkan oleh :citet:`yang2016neural` hanya menawarkan manfaat yang terbatas.

Bandingkan ini dengan basis data (database). Dalam bentuk yang paling sederhana, basis data adalah kumpulan kunci ($k$) dan nilai ($v$). Misalnya, basis data kita $\mathcal{D}$ mungkin terdiri dari pasangan \{("Zhang", "Aston"), ("Lipton", "Zachary"), ("Li", "Mu"), ("Smola", "Alex"), ("Hu", "Rachel"), ("Werness", "Brent")\}, dengan nama belakang sebagai kunci dan nama depan sebagai nilai. Kita bisa melakukan operasi pada $\mathcal{D}$, misalnya dengan query ($q$) yang tepat untuk "Li", yang akan mengembalikan nilai "Mu". Jika ("Li", "Mu") bukan catatan di $\mathcal{D}$, tidak akan ada jawaban yang valid. Jika kita juga mengizinkan pencocokan mendekati, kita akan mendapatkan ("Lipton", "Zachary") sebagai gantinya. Contoh yang cukup sederhana dan sepele ini tetap mengajarkan kita beberapa hal yang berguna:

* Kita bisa merancang query $q$ yang beroperasi pada pasangan ($k$, $v$) sedemikian rupa sehingga tetap valid terlepas dari ukuran basis data.
* Query yang sama dapat menerima jawaban yang berbeda, tergantung pada isi basis data.
* "Kode" yang dijalankan untuk operasi pada ruang keadaan besar (basis data) bisa sangat sederhana (misalnya pencocokan tepat, pencocokan mendekati, top-$k$).
* Tidak perlu mengompresi atau menyederhanakan basis data agar operasi menjadi efektif.

Jelas, kita tidak akan memperkenalkan basis data yang sederhana di sini jika tidak untuk tujuan menjelaskan deep learning. Faktanya, ini mengarah pada salah satu konsep paling menarik yang diperkenalkan dalam deep learning dalam dekade terakhir: *attention mechanism* :cite:`Bahdanau.Cho.Bengio.2014`. Kami akan membahas rincian penggunaannya dalam machine translation nanti. Untuk saat ini, anggap saja berikut ini: misalkan $\mathcal{D} \stackrel{\textrm{def}}{=} \{(\mathbf{k}_1, \mathbf{v}_1), \ldots (\mathbf{k}_m, \mathbf{v}_m)\}$ adalah basis data yang berisi $m$ pasangan *keys* dan *values*. Selain itu, misalkan $\mathbf{q}$ adalah *query*. Maka kita dapat mendefinisikan *attention* terhadap $\mathcal{D}$ sebagai

$$\textrm{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\textrm{def}}{=} \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i,$$
:eqlabel:`eq_attention_pooling`

di mana $\alpha(\mathbf{q}, \mathbf{k}_i) \in \mathbb{R}$ ($i = 1, \ldots, m$) adalah bobot attention skalar. Operasi itu sendiri sering disebut sebagai *attention pooling*. Nama *attention* berasal dari fakta bahwa operasi ini memberikan perhatian khusus pada elemen-elemen di mana bobot $\alpha$ signifikan (yaitu besar). Dengan demikian, attention terhadap $\mathcal{D}$ menghasilkan kombinasi linear dari nilai-nilai yang terkandung dalam basis data. Bahkan, ini mengandung contoh di atas sebagai kasus khusus di mana semua bobot kecuali satu adalah nol. Kita memiliki sejumlah kasus khusus:

* Bobot $\alpha(\mathbf{q}, \mathbf{k}_i)$ bernilai non-negatif. Dalam hal ini, output dari mekanisme attention berada dalam kerucut cembung yang dibentuk oleh nilai-nilai $\mathbf{v}_i$.
* Bobot $\alpha(\mathbf{q}, \mathbf{k}_i)$ membentuk kombinasi cembung, yaitu $\sum_i \alpha(\mathbf{q}, \mathbf{k}_i) = 1$ dan $\alpha(\mathbf{q}, \mathbf{k}_i) \geq 0$ untuk semua $i$. Ini adalah pengaturan yang paling umum dalam deep learning.
* Tepat satu dari bobot $\alpha(\mathbf{q}, \mathbf{k}_i)$ bernilai $1$, sementara yang lainnya adalah $0$. Ini mirip dengan query basis data tradisional.
* Semua bobot bernilai sama, yaitu $\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{1}{m}$ untuk semua $i$. Ini setara dengan rata-rata seluruh basis data, yang juga disebut average pooling dalam deep learning.

Strategi umum untuk memastikan bahwa bobot-bobot tersebut berjumlah $1$ adalah dengan menormalkannya melalui

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{{\sum_j} \alpha(\mathbf{q}, \mathbf{k}_j)}.$$

Secara khusus, untuk memastikan bahwa bobot juga bernilai non-negatif, kita dapat menggunakan eksponensiasi. Ini berarti kita dapat memilih *fungsi apapun* $a(\mathbf{q}, \mathbf{k})$ dan kemudian menerapkan operasi softmax yang digunakan untuk model multinomial melalui

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_j \exp(a(\mathbf{q}, \mathbf{k}_j))}. $$
:eqlabel:`eq_softmax_attention`

Operasi ini tersedia di semua framework deep learning. Operasi ini diferensial dan gradiennya tidak pernah menghilang, yang semuanya merupakan sifat yang diinginkan dalam sebuah model. Perlu dicatat bahwa mekanisme attention yang diperkenalkan di atas bukan satu-satunya pilihan. Misalnya, kita bisa merancang model attention non-differentiable yang dapat dilatih menggunakan metode reinforcement learning :cite:`Mnih.Heess.Graves.ea.2014`. Seperti yang bisa diduga, melatih model semacam itu cukup kompleks. Akibatnya, sebagian besar penelitian attention modern mengikuti kerangka kerja yang diuraikan dalam :numref:`fig_qkv`. Oleh karena itu, kami akan fokus pada mekanisme yang dapat dibedakan dalam keluarga ini.

![Mekanisme attention menghitung kombinasi linear dari values $\mathbf{v}_\mathit{i}$ melalui attention pooling, di mana bobot diturunkan sesuai dengan kompatibilitas antara query $\mathbf{q}$ dan keys $\mathbf{k}_\mathit{i}$.](../img/qkv.svg)
:label:`fig_qkv`

Yang cukup mengesankan adalah bahwa "kode" aktual untuk eksekusi pada set keys dan values, yaitu query, bisa sangat ringkas, meskipun ruang yang dioperasikan cukup signifikan. Ini adalah sifat yang diinginkan untuk sebuah layer jaringan karena tidak memerlukan terlalu banyak parameter untuk dipelajari. Sama mudahnya adalah kenyataan bahwa attention dapat beroperasi pada basis data yang ukurannya arbitrarily besar tanpa perlu mengubah cara operasi attention pooling dilakukan.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input  n=2}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
```

## Visualisasi

Salah satu manfaat dari mekanisme attention adalah bahwa mekanisme ini bisa sangat intuitif, terutama ketika bobotnya non-negatif dan berjumlah $1$. Dalam hal ini, kita mungkin *menginterpretasikan* bobot besar sebagai cara bagi model untuk memilih komponen yang relevan. Meskipun ini merupakan intuisi yang baik, penting untuk diingat bahwa ini hanyalah sebuah *intuisi*. Bagaimanapun, kita mungkin ingin memvisualisasikan efeknya pada sekumpulan keys tertentu ketika menerapkan berbagai queries. Fungsi ini akan berguna nanti.

Oleh karena itu, kita mendefinisikan fungsi `show_heatmaps`. Perlu dicatat bahwa fungsi ini tidak menerima matriks (dari bobot attention) sebagai input, tetapi menerima tensor dengan empat sumbu, yang memungkinkan array dari berbagai queries dan bobot. Akibatnya, input `matrices` memiliki bentuk (jumlah baris untuk tampilan, jumlah kolom untuk tampilan, jumlah queries, jumlah keys). Ini akan berguna nanti ketika kita ingin memvisualisasikan cara kerja yang digunakan untuk merancang Transformers.


```{.python .input  n=17}
%%tab all
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Menampilkan heatmaps dari matriks."""
    d2l.use_svg_display()
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            if tab.selected('pytorch', 'mxnet', 'tensorflow'):
                pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if tab.selected('jax'):
                pcm = ax.imshow(matrix, cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

Sebagai pemeriksaan cepat, mari kita visualisasikan matriks identitas, yang merepresentasikan sebuah kasus di mana bobot attention adalah $1$ hanya ketika query dan key adalah sama.


```{.python .input  n=20}
%%tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

## Ringkasan

Mekanisme attention memungkinkan kita untuk mengagregasi data dari banyak pasangan (key, value). Sampai saat ini, diskusi kita masih cukup abstrak, hanya menjelaskan cara mengumpulkan data. Kita belum menjelaskan dari mana query, key, dan value yang misterius tersebut berasal. Beberapa intuisi mungkin membantu di sini: misalnya, dalam pengaturan regresi, query mungkin sesuai dengan lokasi di mana regresi harus dilakukan. Keys adalah lokasi di mana data sebelumnya diamati dan values adalah nilai-nilai (regresi) itu sendiri. Ini adalah estimator yang dikenal sebagai Nadaraya--Watson :cite:`Nadaraya.1964,Watson.1964` yang akan kita pelajari di bagian berikutnya.

Secara desain, mekanisme attention menyediakan cara *differentiable* (dapat diturunkan) untuk mengontrol di mana jaringan saraf dapat memilih elemen dari sebuah set dan membentuk jumlah tertimbang yang terkait dari representasi.

## Latihan

1. Misalkan Anda ingin mengimplementasikan ulang pencocokan (key, query) secara aproksimasi seperti yang digunakan dalam basis data klasik, fungsi attention mana yang akan Anda pilih?
2. Misalkan fungsi attention diberikan oleh $a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i$ dan $\mathbf{k}_i = \mathbf{v}_i$ untuk $i = 1, \ldots, m$. Notasikan $p(\mathbf{k}_i; \mathbf{q})$ sebagai distribusi probabilitas atas keys ketika menggunakan normalisasi softmax dalam :eqref:`eq_softmax_attention`. Buktikan bahwa $\nabla_{\mathbf{q}} \mathop{\textrm{Attention}}(\mathbf{q}, \mathcal{D}) = \textrm{Cov}_{p(\mathbf{k}_i; \mathbf{q})}[\mathbf{k}_i]$.
3. Desain mesin pencari yang dapat dibedakan menggunakan mekanisme attention.
4. Tinjau desain dari Squeeze and Excitation Networks :cite:`Hu.Shen.Sun.2018` dan interpretasikan melalui perspektif mekanisme attention.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/1596)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1592)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/1710)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18024)
:end_tab:
