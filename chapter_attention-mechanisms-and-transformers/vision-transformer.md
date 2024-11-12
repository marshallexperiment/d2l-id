```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['pytorch', 'jax'])
```

# Transformer untuk Vision
:label:`sec_vision-transformer`

Arsitektur Transformer awalnya diusulkan untuk pembelajaran urutan-ke-urutan (sequence-to-sequence learning), dengan fokus pada penerjemahan mesin. Selanjutnya, Transformer muncul sebagai model pilihan dalam berbagai tugas pemrosesan bahasa alami (NLP) :cite:`Radford.Narasimhan.Salimans.ea.2018,Radford.Wu.Child.ea.2019,brown2020language,Devlin.Chang.Lee.ea.2018,raffel2020exploring`. Namun, di bidang penglihatan komputer (computer vision), arsitektur dominan tetaplah Convolutional Neural Network (CNN) (:numref:`chap_modern_cnn`). Wajar jika peneliti mulai bertanya-tanya apakah mungkin bisa melakukan lebih baik dengan mengadaptasi model Transformer untuk data gambar. Pertanyaan ini memicu minat besar dalam komunitas penglihatan komputer.

Belakangan ini, :citet:`ramachandran2019stand` mengusulkan skema untuk menggantikan konvolusi dengan self-attention. Namun, penggunaan pola-pola khusus dalam attention ini membuatnya sulit untuk menskalakan model pada perangkat akselerator hardware. Kemudian, :citet:`cordonnier2020relationship` membuktikan secara teoritis bahwa self-attention dapat belajar untuk berperilaku mirip dengan konvolusi. Secara empiris, patch $2 \times 2$ diambil dari gambar sebagai input, tetapi ukuran patch yang kecil membuat model ini hanya dapat diterapkan pada data gambar dengan resolusi rendah.

Tanpa batasan khusus pada ukuran patch, *vision Transformers* (ViTs) mengekstrak patch dari gambar dan memasukkannya ke dalam encoder Transformer untuk memperoleh representasi global, yang akhirnya akan diubah untuk klasifikasi :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`. Yang perlu dicatat, Transformer menunjukkan skalabilitas yang lebih baik dibandingkan dengan CNN: ketika melatih model yang lebih besar pada dataset yang lebih besar, vision Transformers mengungguli ResNet dengan margin yang signifikan. Mirip dengan lanskap desain arsitektur jaringan di pemrosesan bahasa alami, Transformer juga menjadi pengubah permainan (game-changer) dalam bidang penglihatan komputer.


```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Model

:numref:`fig_vit` menggambarkan arsitektur model vision Transformer. Arsitektur ini terdiri dari tiga bagian utama: stem yang memecah gambar menjadi patch, tubuh yang berbasis multilayer Transformer encoder, dan head yang mengubah representasi global menjadi label output.

![Arsitektur vision Transformer. Dalam contoh ini, sebuah gambar dibagi menjadi sembilan patch. Token khusus “&lt;cls&gt;” dan sembilan patch gambar yang telah diratakan diubah melalui embedding patch dan $\mathit{n}$ blok Transformer encoder menjadi sepuluh representasi, masing-masing. Representasi dari token “&lt;cls&gt;” selanjutnya diubah menjadi label output.](../img/vit.svg)
:label:`fig_vit`

Misalkan sebuah gambar masukan memiliki tinggi $h$, lebar $w$, dan $c$ kanal. Dengan menentukan tinggi dan lebar patch sebagai $p$, gambar dibagi menjadi urutan $m = hw/p^2$ patch, di mana setiap patch diratakan menjadi vektor dengan panjang $cp^2$. Dengan cara ini, patch gambar dapat diperlakukan serupa dengan token dalam urutan teks oleh Transformer encoder. Token khusus “&lt;cls&gt;” (class) dan $m$ patch gambar yang telah diratakan diproyeksikan secara linear menjadi urutan $m+1$ vektor, yang kemudian dijumlahkan dengan positional embedding yang dapat dipelajari. Multilayer Transformer encoder mengubah $m+1$ vektor masukan menjadi representasi vektor keluaran dengan jumlah dan panjang yang sama. Cara kerjanya persis sama seperti Transformer encoder asli pada :numref:`fig_transformer`, hanya berbeda pada posisi normalisasi. Karena token “&lt;cls&gt;” berinteraksi dengan semua patch gambar melalui self-attention (lihat :numref:`fig_cnn-rnn-self-attention`), representasinya dari keluaran Transformer encoder akan diubah lebih lanjut menjadi label output.

## Patch Embedding

Untuk mengimplementasikan vision Transformer, mari kita mulai dengan patch embedding di :numref:`fig_vit`. Memecah gambar menjadi patch dan memproyeksikan patch yang telah diratakan secara linear dapat disederhanakan menjadi satu operasi konvolusi, di mana ukuran kernel dan ukuran stride sama dengan ukuran patch.


```{.python .input}
%%tab pytorch
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
```

```{.python .input}
%%tab jax
class PatchEmbedding(nn.Module):
    img_size: int = 96
    patch_size: int = 16
    num_hiddens: int = 512

    def setup(self):
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(self.img_size), _make_tuple(self.patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.Conv(self.num_hiddens, kernel_size=patch_size,
                            strides=patch_size, padding='SAME')

    def __call__(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        X = self.conv(X)
        return X.reshape((X.shape[0], -1, X.shape[3]))
```

Pada contoh berikut, kita menggunakan gambar dengan tinggi dan lebar sebesar `img_size` sebagai input, dan patch embedding menghasilkan output sebanyak `(img_size//patch_size)**2` patch yang diproyeksikan secara linear menjadi vektor dengan panjang `num_hiddens`.


```{.python .input}
%%tab pytorch
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.zeros(batch_size, 3, img_size, img_size)
d2l.check_shape(patch_emb(X),
                (batch_size, (img_size//patch_size)**2, num_hiddens))
```

```{.python .input}
%%tab jax
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.zeros((batch_size, img_size, img_size, 3))
output, _ = patch_emb.init_with_output(d2l.get_key(), X)
d2l.check_shape(output, (batch_size, (img_size//patch_size)**2, num_hiddens))
```

## Vision Transformer Encoder
:label:`subsec_vit-encoder`

MLP pada Vision Transformer encoder sedikit berbeda
dari Positionwise FFN pada Transformer encoder asli
(lihat :numref:`subsec_positionwise-ffn`).
Pertama, pada MLP ini, fungsi aktivasi yang digunakan adalah Gaussian Error Linear Unit (GELU),
yang dapat dianggap sebagai versi ReLU yang lebih halus :cite:`Hendrycks.Gimpel.2016`.
Kedua, dropout diterapkan pada output dari setiap layer fully connected dalam MLP untuk regularisasi.


```{.python .input}
%%tab pytorch
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))
```

```{.python .input}
%%tab jax
class ViTMLP(nn.Module):
    mlp_num_hiddens: int
    mlp_num_outputs: int
    dropout: float = 0.5

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(self.mlp_num_hiddens)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        x = nn.Dense(self.mlp_num_outputs)(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        return x
```

Implementasi blok Vision Transformer encoder
mengikuti desain pra-normalisasi seperti pada :numref:`fig_vit`,
di mana normalisasi diterapkan tepat *sebelum* multi-head attention atau MLP.
Sebaliknya, pada post-normalisasi ("add & norm" dalam :numref:`fig_transformer`),
normalisasi diterapkan tepat *setelah* koneksi residual,
pra-normalisasi menghasilkan pelatihan yang lebih efektif atau efisien untuk Transformers :cite:`baevski2018adaptive,wang2019learning,xiong2020layer`.


```{.python .input}
%%tab pytorch
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))
```

```{.python .input}
%%tab jax
class ViTBlock(nn.Module):
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.attention = d2l.MultiHeadAttention(self.num_hiddens, self.num_heads,
                                                self.dropout, self.use_bias)
        self.mlp = ViTMLP(self.mlp_num_hiddens, self.num_hiddens, self.dropout)

    @nn.compact
    def __call__(self, X, valid_lens=None, training=False):
        X = X + self.attention(*([nn.LayerNorm()(X)] * 3),
                               valid_lens, training=training)[0]
        return X + self.mlp(nn.LayerNorm()(X), training=training)
```

Sama seperti pada :numref:`subsec_transformer-encoder`,
tidak ada blok encoder Vision Transformer yang mengubah bentuk inputnya.


```{.python .input}
%%tab pytorch
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X), X.shape)
```

```{.python .input}
%%tab jax
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 48, 8, 0.5)
d2l.check_shape(encoder_blk.init_with_output(d2l.get_key(), X)[0], X.shape)
```

## Menggabungkan Semuanya

Proses forward pass pada Vision Transformers di bawah ini cukup sederhana.
Pertama, gambar input dimasukkan ke dalam instance `PatchEmbedding`,
yang outputnya digabungkan dengan embedding token “<cls>”.
Output tersebut dijumlahkan dengan embedding posisi yang dapat dipelajari sebelum melalui proses dropout.
Kemudian, output tersebut dimasukkan ke dalam Transformer encoder yang menyusun `num_blks` instance dari kelas `ViTBlock`.
Terakhir, representasi token “<cls>” diproyeksikan oleh network head.


```{.python .input}
%%tab pytorch
class ViT(d2l.Classifier):
    """Vision Transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(d2l.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Menambahkan token cls
        # Positional embeddings yang dapat dipelajari
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
```

```{.python .input}
%%tab jax
class ViT(d2l.Classifier):
    """Vision Transformer."""
    img_size: int
    patch_size: int
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    num_blks: int
    emb_dropout: float
    blk_dropout: float
    lr: float = 0.1
    use_bias: bool = False
    num_classes: int = 10
    training: bool = False

    def setup(self):
        self.patch_embedding = PatchEmbedding(self.img_size, self.patch_size,
                                              self.num_hiddens)
        self.cls_token = self.param('cls_token', nn.initializers.zeros,
                                    (1, 1, self.num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Menambahkan token cls
        # Positional embeddings yang dapat dipelajari
        self.pos_embedding = self.param('pos_embed', nn.initializers.normal(),
                                        (1, num_steps, self.num_hiddens))
        self.blks = [ViTBlock(self.num_hiddens, self.mlp_num_hiddens,
                              self.num_heads, self.blk_dropout, self.use_bias)
                    for _ in range(self.num_blks)]
        self.head = nn.Sequential([nn.LayerNorm(), nn.Dense(self.num_classes)])

    @nn.compact
    def __call__(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((jnp.tile(self.cls_token, (X.shape[0], 1, 1)), X), 1)
        X = nn.Dropout(emb_dropout, deterministic=not self.training)(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X, training=self.training)
        return self.head(X[:, 0])
```

## Pelatihan

Pelatihan vision Transformer pada dataset Fashion-MNIST sama seperti cara melatih CNN pada :numref:`chap_modern_cnn`.


```{.python .input}
%%tab all
img_size, patch_size = 96, 16
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, lr)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
trainer.fit(model, data)
```

## Ringkasan dan Diskusi

Anda mungkin telah menyadari bahwa untuk dataset kecil seperti Fashion-MNIST,
vision Transformer yang telah kita implementasikan tidak mengungguli ResNet di :numref:`sec_resnet`.
Pengamatan serupa juga dapat dibuat pada dataset ImageNet (1,2 juta gambar).
Hal ini dikarenakan Transformer *tidak memiliki* prinsip-prinsip yang berguna dalam konvolusi,
seperti translasi invariansi dan lokalitas (:numref:`sec_why-conv`).
Namun, gambarannya berubah ketika melatih model yang lebih besar pada dataset yang lebih besar (misalnya, 300 juta gambar),
di mana vision Transformer mengungguli ResNet dengan margin yang besar dalam klasifikasi gambar, menunjukkan keunggulan intrinsik dari Transformer dalam skalabilitas :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
Pengenalan vision Transformer telah mengubah lanskap desain jaringan untuk pemodelan data gambar.
Mereka segera terbukti efektif pada dataset ImageNet dengan strategi pelatihan data-efisien dari DeiT :cite:`touvron2021training`.
Namun, kompleksitas kuadrat dari self-attention (:numref:`sec_self-attention-and-positional-encoding`)
membuat arsitektur Transformer kurang cocok untuk gambar beresolusi tinggi.
Menuju jaringan backbone yang lebih umum di computer vision,
Swin Transformer menangani kompleksitas komputasi kuadrat terhadap ukuran gambar (:numref:`subsec_cnn-rnn-self-attention`)
dan mengembalikan prior seperti konvolusi,
memperluas penerapan Transformer ke berbagai tugas computer vision di luar klasifikasi gambar dengan hasil terbaik :cite:`liu2021swin`.

## Latihan

1. Bagaimana nilai `img_size` mempengaruhi waktu pelatihan?
2. Alih-alih memproyeksikan representasi token “&lt;cls&gt;” ke output, bagaimana jika Anda memproyeksikan rata-rata representasi patch? Implementasikan perubahan ini dan lihat bagaimana pengaruhnya terhadap akurasi.
3. Bisakah Anda mengubah hiperparameter untuk meningkatkan akurasi vision Transformer?

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/8943)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18032)
:end_tab:
