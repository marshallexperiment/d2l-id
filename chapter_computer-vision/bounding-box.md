# Deteksi Objek dan Kotak Pembatas
:label:`sec_bbox`

Pada bagian sebelumnya (misalnya, :numref:`sec_alexnet`--:numref:`sec_googlenet`),
kami memperkenalkan berbagai model untuk klasifikasi gambar.
Pada tugas klasifikasi gambar,
kita mengasumsikan bahwa hanya ada *satu* objek utama
dalam gambar dan kita hanya fokus pada bagaimana
mengenali kategorinya.
Namun, sering kali ada *beberapa* objek
dalam gambar yang menjadi perhatian.
Kita tidak hanya ingin mengetahui kategorinya, tetapi juga posisi spesifiknya dalam gambar.
Dalam visi komputer, tugas semacam ini disebut sebagai *deteksi objek* (atau *pengakuan objek*).

Deteksi objek telah
banyak diterapkan di berbagai bidang.
Sebagai contoh, kendaraan otonom perlu merencanakan
rute perjalanan
dengan mendeteksi posisi
kendaraan, pejalan kaki, jalan, dan rintangan pada gambar video yang diambil.
Selain itu,
robot dapat menggunakan teknik ini
untuk mendeteksi dan menentukan lokasi objek yang menarik
selama navigasi di lingkungan.
Selain itu,
sistem keamanan
mungkin perlu mendeteksi objek abnormal, seperti penyusup atau bom.

Dalam beberapa bagian berikutnya, kami akan memperkenalkan
beberapa metode pembelajaran mendalam untuk deteksi objek.
Kita akan mulai dengan pengenalan
tentang *posisi* (atau *lokasi*) dari objek.


```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

Kita akan memuat gambar sampel yang akan digunakan pada bagian ini. Kita dapat melihat bahwa ada seekor anjing di sisi kiri gambar dan seekor kucing di sisi kanan.
Mereka adalah dua objek utama dalam gambar ini.



```{.python .input}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## Kotak Pembatas

Dalam deteksi objek,
kita biasanya menggunakan *kotak pembatas* untuk mendeskripsikan lokasi spasial suatu objek.
Kotak pembatas berbentuk persegi panjang, yang ditentukan oleh koordinat $x$ dan $y$ dari sudut kiri atas persegi panjang dan koordinat serupa dari sudut kanan bawah. 
Representasi kotak pembatas lain yang umum digunakan adalah koordinat sumbu $(x, y)$ dari pusat kotak pembatas, serta lebar dan tinggi kotak tersebut.

[**Di sini kami mendefinisikan fungsi untuk mengonversi antara**] kedua (**representasi ini**):
`box_corner_to_center` mengonversi dari representasi dua sudut
ke representasi pusat-lebar-tinggi,
dan `box_center_to_corner` sebaliknya.
Argumen input `boxes` harus berupa tensor dua dimensi dengan
bentuk ($n$, 4), di mana $n$ adalah jumlah kotak pembatas.



```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """Konversi dari (kiri-atas, kanan-bawah) ke (pusat, lebar, tinggi)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """Konversi dari (kiri-atas, kanan-bawah) ke (pusat, lebar, tinggi)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

Kita akan [**mendefinisikan kotak pembatas untuk anjing dan kucing dalam gambar**] berdasarkan informasi koordinat.
Titik asal koordinat dalam gambar
adalah sudut kiri atas gambar, dan ke kanan serta ke bawah merupakan
arah positif dari sumbu $x$ dan $y$, masing-masing.


```{.python .input}
#@tab all
# Di sini `bbox` adalah singkatan dari bounding box
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

Kita dapat memverifikasi kebenaran dari dua fungsi konversi kotak pembatas dengan melakukan konversi dua kali.


```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

Mari kita [**gambar kotak pembatas pada gambar**] untuk memeriksa apakah mereka akurat.
Sebelum menggambar, kita akan mendefinisikan fungsi bantu `bbox_to_rect`. Fungsi ini merepresentasikan kotak pembatas dalam format kotak pembatas yang digunakan oleh paket `matplotlib`.


```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """Mengonversi kotak pembatas ke format matplotlib."""
    # Konversi format kotak pembatas (kiri-atas x, kiri-atas y, kanan-bawah x,
    # kanan-bawah y) ke format matplotlib: ((kiri-atas x,
    # kiri-atas y), lebar, tinggi)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

Setelah menambahkan kotak pembatas pada gambar,
kita dapat melihat bahwa garis besar utama dari kedua objek pada dasarnya berada di dalam kedua kotak tersebut.


```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## Ringkasan

* Deteksi objek tidak hanya mengenali semua objek yang relevan dalam gambar, tetapi juga posisinya. Posisi ini umumnya direpresentasikan dengan kotak pembatas berbentuk persegi panjang.
* Kita dapat melakukan konversi antara dua representasi kotak pembatas yang umum digunakan.

## Latihan

1. Temukan gambar lain dan coba beri label kotak pembatas yang mencakup objeknya. Bandingkan pelabelan kotak pembatas dengan pelabelan kategori: mana yang biasanya memakan waktu lebih lama?
2. Mengapa dimensi terdalam dari argumen input `boxes` pada fungsi `box_corner_to_center` dan `box_center_to_corner` selalu bernilai 4?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1527)
:end_tab:
