# Fully Convolutional Networks
:label:`sec_fcn`

Seperti yang dibahas di :numref:`sec_semantic_segmentation`,
segmentasi semantik
mengklasifikasikan gambar pada tingkat piksel.
Jaringan konvolusi penuh (*fully convolutional network* atau FCN)
menggunakan jaringan saraf konvolusi untuk
mengubah piksel gambar menjadi kelas piksel :cite:`Long.Shelhamer.Darrell.2015`.
Berbeda dengan CNN yang kita temui sebelumnya
untuk klasifikasi gambar 
atau deteksi objek,
jaringan konvolusi penuh
mengubah 
tinggi dan lebar dari peta fitur antara
kembali ke dimensi gambar input:
ini dicapai melalui
lapisan konvolusi transpos
yang diperkenalkan di :numref:`sec_transposed_conv`.
Sebagai hasilnya,
output klasifikasi
dan gambar input
memiliki korespondensi satu-ke-satu 
pada tingkat piksel:
dimensi kanal pada setiap piksel output 
menyimpan hasil klasifikasi
untuk piksel input pada posisi spasial yang sama.



```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
```

## Model

Di sini kami menjelaskan desain dasar dari model *fully convolutional network* (FCN).
Seperti yang ditunjukkan pada :numref:`fig_fcn`,
model ini pertama-tama menggunakan CNN untuk mengekstraksi fitur gambar,
kemudian mengubah jumlah kanal menjadi
jumlah kelas
melalui lapisan konvolusi $1\times 1$,
dan akhirnya mengubah tinggi dan lebar
peta fitur
menjadi sama
dengan gambar input melalui
konvolusi transpos yang diperkenalkan di :numref:`sec_transposed_conv`.
Sebagai hasilnya,
output model memiliki tinggi dan lebar yang sama dengan gambar input,
di mana kanal output berisi kelas yang diprediksi
untuk piksel input pada posisi spasial yang sama.

![Fully convolutional network.](../img/fcn.svg)
:label:`fig_fcn`

Berikutnya, kita [**menggunakan model ResNet-18 yang telah dilatih sebelumnya pada dataset ImageNet untuk mengekstrak fitur gambar**]
dan menyebut instance model ini sebagai `pretrained_net`.
Beberapa lapisan terakhir dari model ini
termasuk lapisan pooling rata-rata global
dan lapisan fully connected:
lapisan-lapisan ini tidak diperlukan
dalam fully convolutional network.


```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
pretrained_net.features[-3:], pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]
```

Selanjutnya, kita [**membuat instance fully convolutional network `net`**].
Model ini menyalin semua lapisan yang telah dilatih sebelumnya dari ResNet-18
kecuali lapisan global average pooling terakhir
dan lapisan fully connected yang paling dekat
dengan output.


```{.python .input}
#@tab mxnet
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*list(pretrained_net.children())[:-2])
```

Diberikan input dengan tinggi 320 dan lebar 480,
propagasi maju dari `net`
mengurangi tinggi dan lebar input menjadi 1/32 dari ukuran aslinya, yaitu menjadi 10 dan 15.



```{.python .input}
#@tab mxnet
X = np.random.uniform(size=(1, 3, 320, 480))
net(X).shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 3, 320, 480))
net(X).shape
```

Selanjutnya, kita [**menggunakan lapisan konvolusi $1\times 1$ untuk mengubah jumlah kanal output menjadi jumlah kelas (21) pada dataset Pascal VOC2012.**]
Akhirnya, kita perlu (**meningkatkan tinggi dan lebar peta fitur sebanyak 32 kali**) agar sesuai kembali dengan tinggi dan lebar gambar input.
Ingat cara menghitung 
bentuk output dari lapisan konvolusi di :numref:`sec_padding`. 
Karena $(320-64+16\times2+32)/32=10$ dan $(480-64+16\times2+32)/32=15$, kita membangun lapisan konvolusi transpos dengan stride $32$, 
dengan
tinggi dan lebar kernel $64$, padding $16$.
Secara umum,
kita dapat melihat bahwa
untuk stride $s$,
padding $s/2$ (dengan asumsi $s/2$ adalah bilangan bulat),
dan tinggi serta lebar kernel $2s$, 
konvolusi transpos akan meningkatkan
tinggi dan lebar input sebanyak $s$ kali.


```{.python .input}
#@tab mxnet
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(
            num_classes, kernel_size=64, padding=16, strides=32))
```

```{.python .input}
#@tab pytorch
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

## [**Inisialisasi Lapisan Konvolusi Transpos**]

Kita sudah tahu bahwa
lapisan konvolusi transpos dapat meningkatkan
tinggi dan lebar
peta fitur.
Dalam pemrosesan gambar, kita mungkin perlu memperbesar
gambar, yaitu *upsampling*.
*Interpolasi bilinear*
adalah salah satu teknik upsampling yang umum digunakan.
Teknik ini juga sering digunakan untuk menginisialisasi lapisan konvolusi transpos.



Untuk menjelaskan interpolasi bilinear,
misalkan 
diberikan sebuah gambar input
kita ingin 
menghitung setiap piksel 
dari gambar output yang telah di-*upsampling*.
Untuk menghitung piksel dari gambar output
pada koordinat $(x, y)$,
pertama map $(x, y)$ ke koordinat $(x', y')$ pada gambar input, misalnya, sesuai dengan rasio ukuran input terhadap ukuran output. 
Perhatikan bahwa nilai $x'$ dan $y'$ yang dipetakan adalah bilangan real. 
Kemudian, cari empat piksel terdekat dengan koordinat
$(x', y')$ pada gambar input. 
Terakhir, piksel dari gambar output pada koordinat $(x, y)$ dihitung berdasarkan keempat piksel terdekat
pada gambar input dan jarak relatif mereka dari $(x', y')$. 

Upsampling menggunakan interpolasi bilinear
dapat diimplementasikan dengan lapisan konvolusi transpos
dengan kernel yang dibangun oleh fungsi `bilinear_kernel` berikut. 
Karena keterbatasan ruang, kami hanya memberikan implementasi fungsi `bilinear_kernel` di bawah ini
tanpa diskusi tentang desain algoritmenya.



```{.python .input}
#@tab mxnet
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)
```

```{.python .input}
#@tab pytorch
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
```

Mari kita [**bereksperimen dengan upsampling menggunakan interpolasi bilinear**] 
yang diimplementasikan oleh lapisan konvolusi transpos. 
Kita membangun lapisan konvolusi transpos yang 
melipatgandakan tinggi dan lebar,
dan menginisialisasi kernel-nya dengan fungsi `bilinear_kernel`.




```{.python .input}
#@tab mxnet
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))
```

```{.python .input}
#@tab pytorch
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));
```

Baca gambar `X` dan tetapkan output upsampling ke `Y`. Untuk mencetak gambar, kita perlu menyesuaikan posisi dimensi kanal.


```{.python .input}
#@tab mxnet
img = image.imread('../img/catdog.jpg')
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)
```

```{.python .input}
#@tab pytorch
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
```

Seperti yang kita lihat, lapisan konvolusi transpos meningkatkan tinggi dan lebar gambar sebanyak dua kali lipat.
Kecuali untuk skala koordinat yang berbeda,
gambar yang diperbesar dengan interpolasi bilinear dan gambar asli yang dicetak di :numref:`sec_bbox` tampak sama.



```{.python .input}
#@tab mxnet
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0));
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);
```

Dalam *fully convolutional network*, kita [**menginisialisasi lapisan konvolusi transpos dengan upsampling menggunakan interpolasi bilinear. Untuk lapisan konvolusi $1\times 1$, kita menggunakan inisialisasi Xavier.**]



```{.python .input}
#@tab mxnet
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
```

```{.python .input}
#@tab pytorch
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);
```

## [**Membaca Dataset**]

Kita membaca
dataset segmentasi semantik
seperti yang diperkenalkan di :numref:`sec_semantic_segmentation`. 
Bentuk gambar output dari pemotongan acak
ditentukan sebagai $320\times 480$: baik tinggi maupun lebar dapat dibagi oleh $32$.


```{.python .input}
#@tab all
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
```

## [**Pelatihan**]

Sekarang kita dapat melatih
*fully convolutional network* yang telah kita bangun.
Fungsi loss dan perhitungan akurasi di sini
tidak berbeda secara mendasar dari klasifikasi gambar pada bab sebelumnya. 
Karena kita menggunakan kanal output dari
lapisan konvolusi transpos untuk
memprediksi kelas untuk setiap piksel,
dimensi kanal ditentukan dalam perhitungan loss.
Selain itu, akurasi dihitung
berdasarkan kebenaran
kelas yang diprediksi untuk semua piksel.


```{.python .input}
#@tab mxnet
num_epochs, lr, wd, devices = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(devices)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## [**Prediksi**]

Saat melakukan prediksi, kita perlu menstandarkan gambar input
di setiap kanal dan mengubah gambar menjadi format input empat dimensi yang diperlukan oleh CNN.



```{.python .input}
#@tab mxnet
def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = np.expand_dims(X.transpose(2, 0, 1), axis=0)
    pred = net(X.as_in_ctx(devices[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```{.python .input}
#@tab pytorch
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

Untuk [**memvisualisasikan kelas yang diprediksi**] dari setiap piksel, kita memetakan kelas yang diprediksi kembali ke warna labelnya dalam dataset.



```{.python .input}
#@tab mxnet
def label2image(pred):
    colormap = np.array(d2l.VOC_COLORMAP, ctx=devices[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]
```

```{.python .input}
#@tab pytorch
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
```

Gambar-gambar dalam dataset uji memiliki ukuran dan bentuk yang bervariasi.
Karena model menggunakan lapisan konvolusi transpos dengan *stride* 32,
ketika tinggi atau lebar gambar input tidak dapat dibagi oleh 32,
tinggi atau lebar output dari
lapisan konvolusi transpos akan menyimpang dari bentuk gambar input.
Untuk mengatasi masalah ini,
kita dapat memotong beberapa area persegi panjang dengan tinggi dan lebar yang merupakan kelipatan bilangan bulat dari 32 dalam gambar,
dan melakukan propagasi maju
pada piksel-piksel di area tersebut secara terpisah.
Perlu diperhatikan bahwa
gabungan dari area persegi panjang ini harus sepenuhnya mencakup gambar input.
Ketika sebuah piksel tercakup oleh beberapa area persegi panjang,
rata-rata dari output konvolusi transpos
di area-area tersebut untuk piksel yang sama
dapat digunakan sebagai input untuk
operasi softmax untuk memprediksi kelas.

Untuk menyederhanakan, kita hanya membaca beberapa gambar uji yang lebih besar,
dan memotong area $320\times480$ untuk prediksi dimulai dari sudut kiri atas gambar.
Untuk gambar-gambar uji ini, kita
mencetak area yang dipotong,
hasil prediksi,
dan ground-truth baris demi baris.




```{.python .input}
#@tab mxnet
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input}
#@tab pytorch
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

## Ringkasan

* Fully convolutional network pertama-tama menggunakan CNN untuk mengekstraksi fitur gambar, kemudian mengubah jumlah kanal menjadi jumlah kelas melalui lapisan konvolusi $1\times 1$, dan akhirnya mengubah tinggi dan lebar peta fitur menjadi sama dengan gambar input melalui konvolusi transpos.
* Dalam fully convolutional network, kita dapat menggunakan upsampling dengan interpolasi bilinear untuk menginisialisasi lapisan konvolusi transpos.

## Latihan

1. Jika kita menggunakan inisialisasi Xavier untuk lapisan konvolusi transpos dalam eksperimen, bagaimana hasilnya berubah?
2. Bisakah Anda lebih meningkatkan akurasi model dengan menyetel hiperparameter?
3. Prediksi kelas semua piksel pada gambar uji.
4. Dalam makalah asli fully convolutional network, output dari beberapa lapisan CNN intermediate juga digunakan :cite:`Long.Shelhamer.Darrell.2015`. Coba implementasikan ide ini.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/377)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1582)
:end_tab:

