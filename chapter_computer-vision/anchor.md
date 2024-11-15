# Anchor Box
:label:`sec_anchor`

Algoritma deteksi objek biasanya
mengambil sampel sejumlah besar wilayah pada gambar input, menentukan apakah wilayah-wilayah ini mengandung
objek yang diinginkan, dan menyesuaikan batas-batas
wilayah tersebut untuk memprediksi
*kotak pembatas ground-truth* dari objek dengan lebih akurat.
Model yang berbeda mungkin mengadopsi
skema sampling wilayah yang berbeda.
Di sini kita memperkenalkan salah satu metode tersebut:
metode ini menghasilkan beberapa kotak pembatas dengan skala dan rasio aspek yang bervariasi yang berpusat di setiap piksel.
Kotak pembatas ini disebut *kotak anchor*.
Kita akan merancang model deteksi objek
berdasarkan kotak anchor di :numref:`sec_ssd`.

Pertama, mari kita modifikasi akurasi pencetakan
agar output lebih ringkas.


```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx

np.set_printoptions(2)  # Simplify printing accuracy
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

torch.set_printoptions(2)  # Simplify printing accuracy
```

## Menghasilkan Banyak Kotak Anchor

Misalkan gambar input memiliki tinggi $h$ dan lebar $w$.
Kita menghasilkan kotak anchor dengan bentuk yang berbeda yang berpusat pada setiap piksel gambar.
Biarkan *skala* menjadi $s\in (0, 1]$ dan
*rasio aspek* (rasio lebar terhadap tinggi) adalah $r > 0$.
Kemudian [**lebar dan tinggi kotak anchor adalah $ws\sqrt{r}$ dan $hs/\sqrt{r}$, masing-masing.**]
Perhatikan bahwa ketika posisi pusat diketahui, sebuah kotak anchor dengan lebar dan tinggi tertentu dapat ditentukan.

Untuk menghasilkan beberapa kotak anchor dengan bentuk yang berbeda,
mari kita tetapkan serangkaian skala
$s_1,\ldots, s_n$ dan
serangkaian rasio aspek $r_1,\ldots, r_m$.
Saat menggunakan semua kombinasi skala dan rasio aspek ini dengan setiap piksel sebagai pusatnya,
gambar input akan memiliki total $whnm$ kotak anchor. Meskipun kotak anchor ini mungkin mencakup semua
kotak pembatas ground-truth, kompleksitas komputasinya mudah menjadi terlalu tinggi.
Dalam praktiknya,
kita hanya dapat (**mempertimbangkan kombinasi
yang mengandung**) $s_1$ atau $r_1$:

(**$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$**)

Artinya, jumlah kotak anchor yang berpusat pada piksel yang sama adalah $n+m-1$. Untuk seluruh gambar input, kita akan menghasilkan total $wh(n+m-1)$ kotak anchor.

Metode di atas untuk menghasilkan kotak anchor diimplementasikan dalam fungsi `multibox_prior` berikut. Kita menentukan gambar input, daftar skala, dan daftar rasio aspek, lalu fungsi ini akan mengembalikan semua kotak anchor.


```{.python .input}
#@tab mxnet
#@save
```python
def multibox_prior(data, sizes, ratios):
    """Menghasilkan kotak anchor dengan bentuk yang berbeda yang berpusat pada setiap piksel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.ctx, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, ctx=device)
    ratio_tensor = d2l.tensor(ratios, ctx=device)
    # Offset diperlukan untuk memindahkan kotak anchor ke pusat piksel. Karena
    # sebuah piksel memiliki tinggi=1 dan lebar=1, kita memilih untuk meng-offset pusat kita sebesar 0,5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Langkah yang diskalakan pada sumbu y
    steps_w = 1.0 / in_width  # Langkah yang diskalakan pada sumbu x

    # Menghasilkan semua titik pusat untuk kotak anchor
    center_h = (d2l.arange(in_height, ctx=device) + offset_h) * steps_h
    center_w = (d2l.arange(in_width, ctx=device) + offset_w) * steps_w
    shift_x, shift_y = d2l.meshgrid(center_w, center_h)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)

    # Menghasilkan `boxes_per_pixel` jumlah tinggi dan lebar yang nantinya
    # digunakan untuk membuat koordinat sudut kotak anchor (xmin, xmax, ymin, ymax)
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        sizes[0] * np.sqrt(ratio_tensor[1:]))) \
                        * in_height / in_width  # Menangani input berbentuk persegi panjang
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))
    # Bagi 2 untuk mendapatkan setengah tinggi dan setengah lebar
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2

    # Setiap titik pusat akan memiliki `boxes_per_pixel` jumlah kotak anchor, jadi
    # menghasilkan grid dari semua pusat kotak anchor dengan pengulangan `boxes_per_pixel`
    out_grid = d2l.stack([shift_x, shift_y, shift_x, shift_y],
                         axis=1).repeat(boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return np.expand_dims(output, axis=0)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_prior(data, sizes, ratios):
    """Menghasilkan kotak anchor dengan bentuk yang berbeda yang berpusat pada setiap piksel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)
    # Offset diperlukan untuk memindahkan kotak anchor ke pusat piksel. Karena
    # sebuah piksel memiliki tinggi=1 dan lebar=1, kita memilih untuk meng-offset pusat kita sebesar 0,5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Langkah yang diskalakan pada sumbu y
    steps_w = 1.0 / in_width  # Langkah yang diskalakan pada sumbu x

    # Menghasilkan semua titik pusat untuk kotak anchor
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Menghasilkan `boxes_per_pixel` jumlah tinggi dan lebar yang nantinya
    # digunakan untuk membuat koordinat sudut kotak anchor (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # Menangani input berbentuk persegi panjang
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Bagi 2 untuk mendapatkan setengah tinggi dan setengah lebar
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # Setiap titik pusat akan memiliki `boxes_per_pixel` jumlah kotak anchor, jadi
    # menghasilkan grid dari semua pusat kotak anchor dengan pengulangan `boxes_per_pixel`
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```

Kita dapat melihat bahwa [**bentuk dari variabel kotak anchor yang dikembalikan `Y`**] adalah
(batch size, jumlah kotak anchor, 4).


```{.python .input}
#@tab mxnet
img = image.imread('../img/catdog.jpg').asnumpy()
h, w = img.shape[:2]

print(h, w)
X = np.random.uniform(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

```{.python .input}
#@tab pytorch
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

Setelah mengubah bentuk variabel kotak anchor `Y` menjadi (tinggi gambar, lebar gambar, jumlah kotak anchor yang berpusat pada piksel yang sama, 4),
kita dapat memperoleh semua kotak anchor yang berpusat pada posisi piksel tertentu.
Berikut ini,
kita [**mengakses kotak anchor pertama yang berpusat pada (250, 250)**]. Kotak ini memiliki empat elemen: koordinat sumbu $(x, y)$ di sudut kiri atas dan koordinat sumbu $(x, y)$ di sudut kanan bawah dari kotak anchor.
Nilai koordinat dari kedua sumbu
dibagi dengan lebar dan tinggi gambar, masing-masing.


```{.python .input}
#@tab all
boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :]
```

Untuk [**menampilkan semua kotak anchor yang berpusat pada satu piksel dalam gambar**],
kita mendefinisikan fungsi `show_bboxes` berikut untuk menggambar beberapa kotak pembatas pada gambar.


```{.python .input}
#@tab all
#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """menunjukkan bounding boxes."""

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

Seperti yang baru saja kita lihat, nilai koordinat dari sumbu $x$ dan $y$ dalam variabel `boxes` telah dibagi dengan lebar dan tinggi gambar, masing-masing.
Saat menggambar kotak anchor,
kita perlu mengembalikan nilai koordinat tersebut ke bentuk aslinya;
oleh karena itu, kita mendefinisikan variabel `bbox_scale` di bawah ini.
Sekarang, kita dapat menggambar semua kotak anchor yang berpusat pada (250, 250) dalam gambar.
Seperti yang bisa Anda lihat, kotak anchor berwarna biru dengan skala 0.75 dan rasio aspek 1
mengelilingi anjing dalam gambar dengan baik.


```{.python .input}
#@tab all
d2l.set_figsize()
bbox_scale = d2l.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```

## [**Intersection over Union (IoU)**]

Kita baru saja menyebutkan bahwa sebuah kotak anchor "mengelilingi" anjing dalam gambar dengan baik.
Jika kotak pembatas ground-truth dari objek diketahui, bagaimana "baik" ini dapat diukur secara kuantitatif?
Secara intuitif, kita dapat mengukur kemiripan antara
kotak anchor dan kotak pembatas ground-truth.
Kita tahu bahwa *indeks Jaccard* dapat mengukur kemiripan antara dua himpunan. Diberikan himpunan $\mathcal{A}$ dan $\mathcal{B}$, indeks Jaccard mereka adalah ukuran dari irisan mereka dibagi dengan ukuran dari gabungan mereka:

$$J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$$

Sebenarnya, kita bisa menganggap area piksel dari kotak pembatas sebagai himpunan piksel.
Dengan cara ini, kita dapat mengukur kemiripan kedua kotak pembatas melalui indeks Jaccard dari himpunan piksel mereka. Untuk dua kotak pembatas, kita biasanya menyebut indeks Jaccard mereka sebagai *intersection over union* (*IoU*), yaitu rasio antara area irisan dengan area gabungan mereka, seperti yang ditunjukkan pada :numref:`fig_iou`.
Nilai IoU berkisar antara 0 hingga 1:
0 berarti dua kotak pembatas tidak saling tumpang tindih sama sekali,
sedangkan 1 menunjukkan bahwa kedua kotak pembatas tersebut sama.

![IoU adalah rasio antara area irisan dengan area gabungan dari dua kotak pembatas.](../img/iou.svg)
:label:`fig_iou`

Untuk sisa bagian ini, kita akan menggunakan IoU untuk mengukur kemiripan antara kotak anchor dengan kotak pembatas ground-truth, dan antara kotak anchor yang berbeda.
Diberikan dua daftar kotak anchor atau kotak pembatas,
fungsi `box_iou` berikut menghitung IoU sepasang
di antara dua daftar ini.


```{.python .input}
#@tab mxnet
#@save
def box_iou(boxes1, boxes2):
    """Menghitung IoU sepasang di antara dua daftar kotak anchor atau kotak pembatas."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Bentuk dari `boxes1`, `boxes2`, `areas1`, `areas2`: (jumlah kotak1, 4),
    # (jumlah kotak2, 4), (jumlah kotak1,), (jumlah kotak2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Bentuk dari `inter_upperlefts`, `inter_lowerrights`, `inters`: (jumlah
    # kotak1, jumlah kotak2, 2)
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # Bentuk dari `inter_areas` dan `union_areas`: (jumlah kotak1, jumlah kotak2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

```{.python .input}
#@tab pytorch
#@save
def box_iou(boxes1, boxes2):
    """Menghitung IoU sepasang di antara dua daftar kotak anchor atau kotak pembatas."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Bentuk dari `boxes1`, `boxes2`, `areas1`, `areas2`: (jumlah kotak1, 4),
    # (jumlah kotak2, 4), (jumlah kotak1,), (jumlah kotak2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Bentuk dari `inter_upperlefts`, `inter_lowerrights`, `inters`: (jumlah
    # kotak1, jumlah kotak2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Bentuk dari `inter_areas` dan `union_areas`: (jumlah kotak1, jumlah kotak2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

## Memberi Label pada Kotak Anchor dalam Data Latih
:label:`subsec_labeling-anchor-boxes`

Dalam dataset pelatihan,
kita menganggap setiap kotak anchor sebagai contoh pelatihan.
Untuk melatih model deteksi objek,
kita membutuhkan label *kelas* dan *offset* untuk setiap kotak anchor,
di mana yang pertama adalah
kelas dari objek yang relevan dengan kotak anchor
dan yang kedua adalah offset
kotak pembatas ground-truth relatif terhadap kotak anchor.
Selama prediksi,
untuk setiap gambar
kita menghasilkan beberapa kotak anchor,
memprediksi kelas dan offset untuk semua kotak anchor,
menyesuaikan posisi mereka sesuai dengan offset yang diprediksi untuk memperoleh kotak pembatas yang diprediksi,
dan akhirnya hanya mengeluarkan kotak pembatas yang
memenuhi kriteria tertentu.

Seperti yang kita ketahui, set pelatihan deteksi objek
datang dengan label untuk
lokasi *kotak pembatas ground-truth*
dan kelas dari objek yang dikelilinginya.
Untuk memberi label pada setiap *kotak anchor* yang dihasilkan,
kita merujuk pada lokasi dan kelas
kotak pembatas ground-truth yang *ditugaskan* terdekat dengan kotak anchor.
Berikut ini,
kami menjelaskan algoritme untuk menetapkan
kotak pembatas ground-truth terdekat dengan kotak anchor.

### [**Menetapkan Kotak Pembatas Ground-Truth ke Kotak Anchor**]

Diberikan sebuah gambar,
misalkan kotak anchor adalah $A_1, A_2, \ldots, A_{n_a}$ dan kotak pembatas ground-truth adalah $B_1, B_2, \ldots, B_{n_b}$, di mana $n_a \geq n_b$.
Mari kita definisikan matriks $\mathbf{X} \in \mathbb{R}^{n_a \times n_b}$, yang elemennya $x_{ij}$ pada baris $i^\textrm{th}$ dan kolom $j^\textrm{th}$ adalah IoU antara kotak anchor $A_i$ dan kotak pembatas ground-truth $B_j$. Algoritme ini terdiri dari langkah-langkah berikut:

1. Temukan elemen terbesar dalam matriks $\mathbf{X}$ dan tandai indeks baris dan kolomnya sebagai $i_1$ dan $j_1$, masing-masing. Kemudian, kotak pembatas ground-truth $B_{j_1}$ diberikan ke kotak anchor $A_{i_1}$. Ini cukup intuitif karena $A_{i_1}$ dan $B_{j_1}$ adalah pasangan yang paling dekat di antara semua pasangan kotak anchor dan kotak pembatas ground-truth. Setelah penetapan pertama, abaikan semua elemen di baris ${i_1}^\textrm{th}$ dan kolom ${j_1}^\textrm{th}$ dalam matriks $\mathbf{X}$.
2. Temukan elemen terbesar di sisa matriks $\mathbf{X}$ dan tandai indeks baris dan kolomnya sebagai $i_2$ dan $j_2$, masing-masing. Kami menetapkan kotak pembatas ground-truth $B_{j_2}$ ke kotak anchor $A_{i_2}$ dan mengabaikan semua elemen di baris ${i_2}^\textrm{th}$ dan kolom ${j_2}^\textrm{th}$ dalam matriks $\mathbf{X}$.
3. Pada titik ini, elemen di dua baris dan dua kolom dalam matriks $\mathbf{X}$ telah diabaikan. Kami melanjutkan hingga semua elemen di $n_b$ kolom dalam matriks $\mathbf{X}$ diabaikan. Pada saat ini, kami telah menetapkan sebuah kotak pembatas ground-truth ke masing-masing dari $n_b$ kotak anchor.
4. Hanya telusuri kotak anchor yang tersisa sebanyak $n_a - n_b$. Misalnya, diberikan sebuah kotak anchor $A_i$, temukan kotak pembatas ground-truth $B_j$ dengan IoU terbesar dengan $A_i$ di seluruh baris $i^\textrm{th}$ matriks $\mathbf{X}$, dan tetapkan $B_j$ ke $A_i$ hanya jika IoU ini lebih besar dari ambang batas yang telah ditentukan.

Mari kita ilustrasikan algoritme di atas menggunakan contoh konkret.
Seperti yang ditunjukkan dalam :numref:`fig_anchor_label` (kiri), dengan asumsi bahwa nilai maksimum dalam matriks $\mathbf{X}$ adalah $x_{23}$, kita menetapkan kotak pembatas ground-truth $B_3$ ke kotak anchor $A_2$.
Kemudian, kita mengabaikan semua elemen di baris 2 dan kolom 3 dari matriks, temukan nilai terbesar $x_{71}$ pada elemen yang tersisa (daerah berbayang), dan tetapkan kotak pembatas ground-truth $B_1$ ke kotak anchor $A_7$.
Selanjutnya, seperti yang ditunjukkan pada :numref:`fig_anchor_label` (tengah), abaikan semua elemen di baris 7 dan kolom 1 dari matriks, temukan nilai terbesar $x_{54}$ pada elemen yang tersisa (daerah berbayang), dan tetapkan kotak pembatas ground-truth $B_4$ ke kotak anchor $A_5$.
Akhirnya, seperti yang ditunjukkan pada :numref:`fig_anchor_label` (kanan), abaikan semua elemen di baris 5 dan kolom 4 dari matriks, temukan nilai terbesar $x_{92}$ pada elemen yang tersisa (daerah berbayang), dan tetapkan kotak pembatas ground-truth $B_2$ ke kotak anchor $A_9$.
Setelah itu, kita hanya perlu menelusuri
kotak anchor yang tersisa $A_1, A_3, A_4, A_6, A_8$ dan menentukan apakah akan menetapkan kotak pembatas ground-truth sesuai dengan ambang batas.

![Menetapkan kotak pembatas ground-truth ke kotak anchor.](../img/anchor-label.svg)
:label:`fig_anchor_label`

Algoritme ini diimplementasikan dalam fungsi `assign_anchor_to_bbox` berikut.


```{.python .input}
#@tab mxnet
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Menetapkan kotak pembatas ground-truth terdekat ke kotak anchor."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Elemen x_ij di baris i dan kolom j adalah IoU dari
    # kotak anchor i dan kotak pembatas ground-truth j
    jaccard = box_iou(anchors, ground_truth)
    # Inisialisasi tensor untuk menyimpan kotak pembatas ground-truth yang ditugaskan
    # untuk setiap kotak anchor
    anchors_bbox_map = np.full((num_anchors,), -1, dtype=np.int32, ctx=device)
    # Menetapkan kotak pembatas ground-truth sesuai dengan ambang batas
    max_ious, indices = np.max(jaccard, axis=1), np.argmax(jaccard, axis=1)
    anc_i = np.nonzero(max_ious >= iou_threshold)[0]
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = np.full((num_anchors,), -1)
    row_discard = np.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = np.argmax(jaccard)  # Temukan IoU terbesar
        box_idx = (max_idx % num_gt_boxes).astype('int32')
        anc_idx = (max_idx / num_gt_boxes).astype('int32')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

```{.python .input}
#@tab pytorch
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Menetapkan kotak pembatas ground-truth terdekat ke kotak anchor."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Elemen x_ij di baris i dan kolom j adalah IoU dari
    # kotak anchor i dan kotak pembatas ground-truth j
    jaccard = box_iou(anchors, ground_truth)
    # Inisialisasi tensor untuk menyimpan kotak pembatas ground-truth yang ditugaskan
    # untuk setiap kotak anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Menetapkan kotak pembatas ground-truth sesuai dengan ambang batas
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Temukan IoU terbesar
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

### Memberi Label pada Kelas dan Offset

Sekarang kita dapat memberi label pada kelas dan offset untuk setiap kotak anchor. Misalkan sebuah kotak anchor $A$ ditetapkan ke sebuah kotak pembatas ground-truth $B$.
Di satu sisi,
kelas dari kotak anchor $A$ akan
diberi label yang sama dengan $B$.
Di sisi lain,
offset dari kotak anchor $A$
akan diberi label sesuai dengan
posisi relatif antara
koordinat pusat $B$ dan $A$
serta ukuran relatif antara
kedua kotak ini.
Diberikan posisi dan ukuran yang berbeda-beda pada setiap kotak dalam dataset,
kita dapat menerapkan transformasi
pada posisi dan ukuran relatif ini
yang dapat menghasilkan distribusi offset yang lebih merata dan lebih mudah diprediksi.
Berikut ini adalah transformasi umum yang digunakan.
[**Diberikan koordinat pusat dari $A$ dan $B$ masing-masing sebagai $(x_a, y_a)$ dan $(x_b, y_b)$,
lebar mereka sebagai $w_a$ dan $w_b$,
dan tinggi mereka sebagai $h_a$ dan $h_b$.
Kita dapat memberi label offset $A$ sebagai

$$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),$$
**]
dengan nilai default konstanta $\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x=\sigma_y=0.1$, dan $\sigma_w=\sigma_h=0.2$.
Transformasi ini diimplementasikan di bawah ini dalam fungsi `offset_boxes`.


```{.python .input}
#@tab all
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transformasi untuk Offset Kotak Anchor."""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset
```

Jika sebuah kotak anchor tidak ditetapkan ke kotak pembatas ground-truth, kita cukup memberi label kelas dari kotak anchor tersebut sebagai "background" atau latar belakang.
Kotak anchor yang kelasnya adalah background sering disebut sebagai kotak anchor *negatif*,
dan sisanya disebut kotak anchor *positif*.
Kita mengimplementasikan fungsi `multibox_target` berikut
untuk [**memberi label kelas dan offset pada kotak anchor**] (argumen `anchors`) menggunakan kotak pembatas ground-truth (argumen `labels`).
Fungsi ini menetapkan kelas background ke nilai nol dan menambah indeks kelas baru dengan satu.


```{.python .input}
#@tab mxnet
#@save
def multibox_target(anchors, labels):
    """Memberi label pada kotak anchor menggunakan kotak pembatas ground-truth."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.ctx, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = np.tile((np.expand_dims((anchors_bbox_map >= 0),
                                            axis=-1)), (1, 4)).astype('int32')
        # Inisialisasi label kelas dan koordinat kotak pembatas yang ditugaskan dengan nol
        class_labels = d2l.zeros(num_anchors, dtype=np.int32, ctx=device)
        assigned_bb = d2l.zeros((num_anchors, 4), dtype=np.float32,
                                ctx=device)
        # Memberi label kelas kotak anchor menggunakan kotak pembatas ground-truth yang ditugaskan.
        # Jika sebuah kotak anchor tidak ditugaskan apa pun, kita label kelasnya sebagai background (nilai tetap nol)
        indices_true = np.nonzero(anchors_bbox_map >= 0)[0]
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].astype('int32') + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Transformasi offset
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = d2l.stack(batch_offset)
    bbox_mask = d2l.stack(batch_mask)
    class_labels = d2l.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_target(anchors, labels):
    """Memberi label pada kotak anchor menggunakan kotak pembatas ground-truth."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # Inisialisasi label kelas dan koordinat kotak pembatas yang ditugaskan dengan nol
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # Memberi label kelas kotak anchor menggunakan kotak pembatas ground-truth yang ditugaskan.
        # Jika sebuah kotak anchor tidak ditugaskan apa pun, kita label kelasnya sebagai background (nilai tetap nol)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Transformasi offset
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

### Contoh

Mari kita ilustrasikan pelabelan kotak anchor
melalui contoh konkret.
Kita mendefinisikan kotak pembatas ground-truth untuk anjing dan kucing dalam gambar yang dimuat,
di mana elemen pertama adalah kelas (0 untuk anjing dan 1 untuk kucing) dan empat elemen sisanya adalah
koordinat sumbu $(x, y)$
di sudut kiri atas dan sudut kanan bawah
(dengan rentang antara 0 dan 1).
Kita juga membangun lima kotak anchor yang akan diberi label
menggunakan koordinat
sudut kiri atas dan sudut kanan bawah:
$A_0, \ldots, A_4$ (indeks dimulai dari 0).
Kemudian kita [**plot kotak pembatas ground-truth 
dan kotak anchor ini 
pada gambar.**]


```{.python .input}
#@tab all
ground_truth = d2l.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = d2l.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
```

Dengan menggunakan fungsi `multibox_target` yang telah didefinisikan di atas,
kita dapat [**memberi label pada kelas dan offset
kotak-kotak anchor ini berdasarkan
kotak pembatas ground-truth**] untuk anjing dan kucing.
Dalam contoh ini, indeks untuk
kelas background, anjing, dan kucing
adalah 0, 1, dan 2, secara berurutan.
Di bawah ini, kita menambahkan satu dimensi untuk contoh kotak anchor dan kotak pembatas ground-truth.


```{.python .input}
#@tab mxnet
labels = multibox_target(np.expand_dims(anchors, axis=0),
                         np.expand_dims(ground_truth, axis=0))
```

```{.python .input}
#@tab pytorch
labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
```

Hasil yang dikembalikan oleh fungsi terdiri dari tiga item, yang semuanya dalam format tensor.
Item ketiga berisi kelas yang diberi label untuk kotak anchor yang diinput.

Mari kita analisis label kelas yang dikembalikan berdasarkan
posisi kotak anchor dan kotak pembatas ground-truth pada gambar.
Pertama, di antara semua pasangan kotak anchor
dan kotak pembatas ground-truth,
IoU antara kotak anchor $A_4$ dan kotak pembatas ground-truth untuk kucing adalah yang terbesar.
Maka, kelas dari $A_4$ diberi label sebagai kucing.
Setelah mengeluarkan
pasangan yang mengandung $A_4$ atau kotak pembatas ground-truth untuk kucing, di antara pasangan yang tersisa,
pasangan antara kotak anchor $A_1$ dan kotak pembatas ground-truth untuk anjing memiliki IoU terbesar.
Jadi, kelas dari $A_1$ diberi label sebagai anjing.
Selanjutnya, kita perlu menelusuri tiga kotak anchor yang belum diberi label: $A_0$, $A_2$, dan $A_3$.
Untuk $A_0$,
kelas dari kotak pembatas ground-truth dengan IoU terbesar adalah anjing,
tetapi IoU ini berada di bawah ambang batas yang telah ditentukan (0,5),
sehingga kelasnya diberi label sebagai background;
untuk $A_2$,
kelas dari kotak pembatas ground-truth dengan IoU terbesar adalah kucing dan IoU melebihi ambang batas, sehingga kelasnya diberi label sebagai kucing;
untuk $A_3$,
kelas dari kotak pembatas ground-truth dengan IoU terbesar adalah kucing, tetapi nilainya di bawah ambang batas, sehingga kelasnya diberi label sebagai background.


```{.python .input}
#@tab all
labels[2]
```

Item kedua yang dikembalikan adalah variabel mask dengan bentuk (ukuran batch, empat kali jumlah kotak anchor).
Setiap empat elemen dalam variabel mask
sesuai dengan empat nilai offset dari setiap kotak anchor.
Karena kita tidak memperhitungkan deteksi background,
offset dari kelas negatif ini seharusnya tidak memengaruhi fungsi objektif.
Melalui perkalian elemen demi elemen, nilai nol dalam variabel mask akan menyaring offset kelas negatif sebelum menghitung fungsi objektif.


```{.python .input}
#@tab all
labels[1]
```

Item pertama yang dikembalikan berisi empat nilai offset yang diberi label untuk setiap kotak anchor.
Perhatikan bahwa offset dari kotak anchor dengan kelas negatif diberi label sebagai nol.



```{.python .input}
#@tab all
labels[0]
```

## Memprediksi Kotak Pembatas dengan Non-Maximum Suppression
:label:`subsec_predicting-bounding-boxes-nms`

Selama prediksi,
kita menghasilkan beberapa kotak anchor untuk gambar dan memprediksi kelas serta offset untuk masing-masing kotak tersebut.
Sebuah *kotak pembatas yang diprediksi*
diperoleh berdasarkan
kotak anchor dengan offset yang diprediksi.
Berikut ini, kita mengimplementasikan fungsi `offset_inverse`
yang menerima kotak anchor dan
prediksi offset sebagai input dan [**menerapkan transformasi offset inverse untuk
mengembalikan koordinat kotak pembatas yang diprediksi**].




```{.python .input}
#@tab all
#@save
def offset_inverse(anchors, offset_preds):
    """Memprediksi Kotak Pembatas Berdasarkan Kotak Anchor dengan Offset yang Diprediksi."""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = d2l.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = d2l.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
```

Ketika terdapat banyak kotak anchor,
banyak kotak pembatas yang diprediksi (dengan tumpang tindih yang signifikan)
dapat dihasilkan untuk mengelilingi objek yang sama.
Untuk menyederhanakan output,
kita dapat menggabungkan kotak pembatas yang diprediksi serupa
yang termasuk ke dalam objek yang sama
dengan menggunakan *non-maximum suppression* (NMS).

Berikut adalah cara kerja non-maximum suppression:
Untuk sebuah kotak pembatas yang diprediksi $B$,
model deteksi objek menghitung probabilitas prediksi
untuk setiap kelas.
Diberikan $p$ sebagai probabilitas prediksi terbesar,
kelas yang sesuai dengan probabilitas ini adalah kelas yang diprediksi untuk $B$.
Secara khusus, kita menyebut $p$ sebagai *confidence* (skor) dari kotak pembatas yang diprediksi $B$.
Pada gambar yang sama,
semua kotak pembatas yang diprediksi sebagai non-background
diurutkan berdasarkan confidence secara menurun
untuk menghasilkan daftar $L$.
Kemudian kita memanipulasi daftar $L$ yang telah diurutkan dalam langkah-langkah berikut:

1. Pilih kotak pembatas (_Bounding Box_) yang diprediksi $B_1$ dengan confidence tertinggi dari $L$ sebagai basis dan hapus semua kotak pembatas yang diprediksi non-basis yang memiliki IoU dengan $B_1$ melebihi ambang batas yang telah ditentukan $\epsilon$ dari $L$. Pada tahap ini, $L$ menyimpan kotak pembatas yang diprediksi dengan confidence tertinggi tetapi menghapus yang lain yang terlalu mirip dengannya. Intinya, kotak dengan skor confidence *non-maximum* akan *disuppresed*.
2. Pilih kotak pembatas (_Bounding Box_) yang diprediksi $B_2$ dengan confidence tertinggi kedua dari $L$ sebagai basis lain dan hapus semua kotak pembatas yang diprediksi non-basis yang memiliki IoU dengan $B_2$ melebihi $\epsilon$ dari $L$.
3. Ulangi proses di atas hingga semua kotak pembatas yang diprediksi (_Bounding Box_) di $L$ telah digunakan sebagai basis. Pada titik ini, IoU dari setiap pasangan kotak pembatas yang diprediksi di $L$ berada di bawah ambang batas $\epsilon$; sehingga, tidak ada pasangan yang terlalu mirip satu sama lain.
4. Output semua kotak pembatas yang diprediksi dalam daftar $L$.

[**Fungsi `nms` berikut mengurutkan skor confidence secara menurun dan mengembalikan indeksnya.**]


```{.python .input}
#@tab mxnet
#@save
def nms(boxes, scores, iou_threshold):
    """Mengurutkan Skor Confidence dari bounding box yang Diprediksi."""
    B = scores.argsort()[::-1]
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.size > 0:
        i = B[0]
        keep.append(i)
        if B.size == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = np.nonzero(iou <= iou_threshold)[0]
        B = B[inds + 1]
    return np.array(keep, dtype=np.int32, ctx=boxes.ctx)
```

```{.python .input}
#@tab pytorch
#@save
def nms(boxes, scores, iou_threshold):
    """Mengurutkan Skor Confidence dari bounding box yang Diprediksi."""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return d2l.tensor(keep, device=boxes.device)
```

Kami mendefinisikan fungsi `multibox_detection` berikut untuk [**menerapkan non-maximum suppression (NMS) pada kotak pembatas yang diprediksi**].
Jangan khawatir jika implementasinya terlihat sedikit rumit: kami akan menunjukkan cara kerjanya dengan contoh konkret segera setelah implementasi.


```{.python .input}
#@tab mxnet
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """prediksi bounding boxes menggunakan non-maximum suppression."""
    device, batch_size = cls_probs.ctx, cls_probs.shape[0]
    anchors = np.squeeze(anchors, axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = np.max(cls_prob[1:], 0), np.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Temukan semua indeks yang bukan `keep` dan tetapkan kelasnya sebagai background
        all_idx = np.arange(num_anchors, dtype=np.int32, ctx=device)
        combined = d2l.concat((keep, all_idx))
        unique, counts = np.unique(combined, return_counts=True)
        non_keep = unique[counts == 1]
        all_id_sorted = d2l.concat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted].astype('float32')
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Di sini, `pos_threshold` adalah ambang batas untuk prediksi positif (non-background)
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = d2l.concat((np.expand_dims(class_id, axis=1),
                                np.expand_dims(conf, axis=1),
                                predicted_bb), axis=1)
        out.append(pred_info)
    return d2l.stack(out)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """prediksi bounding boxes menggunakan non-maximum suppression."""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Temukan semua indeks yang bukan `keep` dan tetapkan kelasnya sebagai background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
       # Di sini, `pos_threshold` adalah ambang batas untuk prediksi positif (non-background)
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return d2l.stack(out)
```

Sekarang mari kita [**menerapkan implementasi di atas pada contoh konkret dengan empat kotak anchor**].
Untuk kesederhanaan, kita mengasumsikan bahwa
semua offset yang diprediksi adalah nol.
Ini berarti bahwa kotak pembatas yang diprediksi adalah kotak anchor.
Untuk setiap kelas di antara background, anjing, dan kucing,
kita juga mendefinisikan kemungkinan prediksinya.


```{.python .input}
#@tab all
anchors = d2l.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = d2l.tensor([0] * d2l.size(anchors))
cls_probs = d2l.tensor([[0] * 4,  # Predicted background likelihood 
                      [0.9, 0.8, 0.7, 0.1],  # Predicted dog likelihood 
                      [0.1, 0.2, 0.3, 0.9]])  # Predicted cat likelihood
```

Kita dapat [**memvisualisasikan kotak pembatas yang diprediksi ini beserta confidence-nya pada gambar.**]



```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
```

Sekarang kita dapat memanggil fungsi `multibox_detection`
untuk melakukan non-maximum suppression,
dengan ambang batas yang disetel ke 0,5.
Perhatikan bahwa kita menambahkan
dimensi untuk contoh dalam input tensor.

Kita dapat melihat bahwa [**bentuk dari hasil yang dikembalikan**] adalah
(ukuran batch, jumlah kotak anchor, 6).
Enam elemen dalam dimensi terdalam
memberikan informasi keluaran untuk kotak pembatas yang diprediksi yang sama.
Elemen pertama adalah indeks kelas yang diprediksi, yang dimulai dari 0 (0 adalah anjing dan 1 adalah kucing). Nilai -1 menunjukkan background atau penghapusan dalam non-maximum suppression.
Elemen kedua adalah confidence dari kotak pembatas yang diprediksi.
Empat elemen sisanya adalah koordinat sumbu $(x, y)$ dari sudut kiri atas dan 
sudut kanan bawah dari kotak pembatas yang diprediksi, masing-masing (rentang antara 0 dan 1).



```{.python .input}
#@tab mxnet
output = multibox_detection(np.expand_dims(cls_probs, axis=0),
                            np.expand_dims(offset_preds, axis=0),
                            np.expand_dims(anchors, axis=0),
                            nms_threshold=0.5)
output
```

```{.python .input}
#@tab pytorch
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
output
```

Setelah menghapus kotak pembatas yang diprediksi dengan kelas -1,
kita dapat [**mengeluarkan kotak pembatas akhir yang dipertahankan oleh non-maximum suppression**].



```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
for i in d2l.numpy(output[0]):
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [d2l.tensor(i[2:]) * bbox_scale], label)
```

Dalam praktiknya, kita dapat menghapus kotak pembatas yang diprediksi dengan confidence yang lebih rendah bahkan sebelum melakukan non-maximum suppression, sehingga mengurangi komputasi dalam algoritme ini.
Kita juga dapat melakukan post-processing pada hasil non-maximum suppression, misalnya dengan hanya mempertahankan
hasil dengan confidence lebih tinggi pada output akhir.

## Ringkasan

* Kita menghasilkan kotak anchor dengan bentuk yang berbeda yang berpusat di setiap piksel gambar.
* Intersection over union (IoU), juga dikenal sebagai indeks Jaccard, mengukur kemiripan dua kotak pembatas. Ini adalah rasio antara area irisan dan area gabungan mereka.
* Dalam set pelatihan, kita memerlukan dua jenis label untuk setiap kotak anchor. Salah satunya adalah kelas dari objek yang relevan dengan kotak anchor dan yang lainnya adalah offset dari kotak pembatas ground-truth relatif terhadap kotak anchor.
* Selama prediksi, kita dapat menggunakan non-maximum suppression (NMS) untuk menghapus kotak pembatas yang diprediksi yang serupa, sehingga menyederhanakan output.

## Latihan

1. Ubah nilai `sizes` dan `ratios` dalam fungsi `multibox_prior`. Apa perubahan yang terjadi pada kotak anchor yang dihasilkan?
2. Konstruksi dan visualisasikan dua kotak pembatas dengan IoU sebesar 0,5. Bagaimana mereka saling tumpang tindih?
3. Modifikasi variabel `anchors` di :numref:`subsec_labeling-anchor-boxes` dan :numref:`subsec_predicting-bounding-boxes-nms`. Bagaimana hasilnya berubah?
4. Non-maximum suppression adalah algoritme greedy yang menekan kotak pembatas yang diprediksi dengan *menghapus* kotak tersebut. Apakah mungkin bahwa beberapa kotak yang dihapus sebenarnya berguna? Bagaimana algoritme ini bisa dimodifikasi untuk menekan secara *lembut*? Anda dapat merujuk ke Soft-NMS :cite:`Bodla.Singh.Chellappa.ea.2017`.
5. Alih-alih dibuat secara manual, bisakah non-maximum suppression dipelajari?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/370)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1603)
:end_tab:
