# Fine-Tuning
:label:`sec_fine_tuning`

Pada bab sebelumnya, kita telah membahas bagaimana melatih model pada dataset pelatihan Fashion-MNIST yang hanya memiliki 60.000 gambar. Kita juga telah mendeskripsikan ImageNet, dataset gambar skala besar yang paling banyak digunakan di dunia akademik, yang memiliki lebih dari 10 juta gambar dan 1000 objek. Namun, ukuran dataset yang biasanya kita temui berada di antara kedua dataset tersebut.

Misalnya, jika kita ingin mengenali berbagai jenis kursi dari gambar, lalu merekomendasikan tautan pembelian kepada pengguna.
Salah satu metode yang mungkin adalah dengan terlebih dahulu mengidentifikasi 100 jenis kursi yang umum, mengambil 1000 gambar dari sudut yang berbeda untuk setiap kursi, dan kemudian melatih model klasifikasi pada dataset gambar yang dikumpulkan.
Meskipun dataset kursi ini mungkin lebih besar daripada dataset Fashion-MNIST, jumlah contoh yang dimiliki masih kurang dari sepersepuluh dari jumlah yang ada di ImageNet.
Hal ini dapat menyebabkan *overfitting* pada model yang lebih kompleks yang cocok untuk ImageNet ketika digunakan pada dataset kursi tersebut.
Selain itu, karena jumlah contoh pelatihan yang terbatas, akurasi model yang dilatih mungkin tidak memenuhi persyaratan praktis.

Untuk mengatasi masalah di atas, solusi yang jelas adalah mengumpulkan lebih banyak data.
Namun, mengumpulkan dan memberi label data dapat memakan banyak waktu dan biaya.
Misalnya, dalam rangka mengumpulkan dataset ImageNet, para peneliti telah menghabiskan jutaan dolar dari dana penelitian. 
Meskipun biaya pengumpulan data saat ini telah berkurang secara signifikan, biaya tersebut masih tidak bisa diabaikan.

Solusi lain adalah dengan menerapkan *transfer learning* untuk mentransfer pengetahuan yang dipelajari dari *source dataset* ke *target dataset*. 
Sebagai contoh, meskipun sebagian besar gambar dalam dataset ImageNet tidak ada hubungannya dengan kursi, model yang dilatih pada dataset ini mungkin dapat mengekstraksi fitur gambar yang lebih umum, yang dapat membantu mengenali tepi, tekstur, bentuk, dan komposisi objek.
Fitur-fitur serupa ini mungkin juga efektif untuk mengenali kursi.

## Langkah-langkah

Pada bagian ini, kita akan memperkenalkan teknik umum dalam *transfer learning*, yaitu *fine-tuning*. Seperti yang ditunjukkan pada :numref:`fig_finetune`, *fine-tuning* terdiri dari empat langkah berikut:

1. Melatih model jaringan saraf, yaitu *source model*, pada *source dataset* (misalnya, dataset ImageNet).
2. Membuat model jaringan saraf baru, yaitu *target model*. Ini menyalin semua desain model dan parameter dari *source model* kecuali layer output. Kita mengasumsikan bahwa parameter model ini mengandung pengetahuan yang dipelajari dari *source dataset* dan pengetahuan ini juga dapat diterapkan pada *target dataset*. Kita juga mengasumsikan bahwa layer output dari *source model* sangat terkait dengan label dari *source dataset*, sehingga tidak digunakan pada *target model*.
3. Menambahkan layer output ke *target model*, di mana jumlah outputnya sesuai dengan jumlah kategori di *target dataset*. Kemudian, menginisialisasi parameter model pada layer ini secara acak.
4. Melatih *target model* pada *target dataset*, seperti dataset kursi. Layer output akan dilatih dari awal, sementara parameter dari semua layer lainnya disesuaikan (*fine-tuned*) berdasarkan parameter dari *source model*.

![Fine tuning.](../img/finetune.svg)
:label:`fig_finetune`

Ketika *target dataset* jauh lebih kecil dari *source dataset*, *fine-tuning* membantu meningkatkan kemampuan generalisasi model.

## Pengenalan Hot Dog

Mari kita demonstrasikan *fine-tuning* melalui contoh konkret: pengenalan hot dog.
Kita akan *fine-tune* model ResNet pada dataset kecil, yang sebelumnya telah dilatih pada dataset ImageNet.
Dataset kecil ini terdiri dari ribuan gambar yang berisi atau tidak berisi hot dog.
Kita akan menggunakan model yang sudah disesuaikan untuk mengenali hot dog dari gambar.


```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from torch import nn
import torch
import torchvision
import os
```

### Membaca Dataset

[**Dataset hot dog yang kita gunakan diambil dari gambar yang tersedia secara online**].
Dataset ini terdiri dari 1400 gambar *positive class* yang mengandung hot dog, dan sebanyak gambar kelas negatif yang mengandung makanan lain.
Sebanyak 1000 gambar dari kedua kelas digunakan untuk pelatihan, dan sisanya digunakan untuk pengujian.

Setelah mengekstrak dataset yang diunduh, kita akan mendapatkan dua folder `hotdog/train` dan `hotdog/test`. Kedua folder ini memiliki subfolder `hotdog` dan `not-hotdog`, yang masing-masing berisi gambar dari kelas yang sesuai.



```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
```

Kita membuat dua instance untuk membaca semua file gambar yang terdapat pada dataset pelatihan dan pengujian secara berturut-turut.


```{.python .input}
#@tab mxnet
train_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'train'))
test_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'test'))
```

```{.python .input}
#@tab pytorch
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
```

Di bawah ini ditampilkan 8 contoh positif pertama dan 8 gambar negatif terakhir. Seperti yang dapat dilihat, [**gambar-gambar tersebut bervariasi dalam ukuran dan rasio aspek**].



```{.python .input}
#@tab all
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
```

Selama pelatihan, kita terlebih dahulu memotong area acak dengan ukuran dan rasio aspek yang juga acak dari gambar, kemudian memperbesar area ini menjadi gambar input berukuran $224 \times 224$. 

Selama pengujian, kita memperbesar baik tinggi maupun lebar gambar menjadi 256 piksel, lalu memotong area tengah berukuran $224 \times 224$ sebagai input. 

Selain itu, untuk tiga saluran warna RGB (merah, hijau, dan biru), kita *standarisasi* nilainya saluran per saluran. Secara konkret, nilai rata-rata dari setiap saluran dikurangkan dari setiap nilai di saluran tersebut dan kemudian hasilnya dibagi dengan standar deviasi dari saluran tersebut.


[~~Augmentasi Data~~]

```{.python .input}
#@tab mxnet
# Tentukan nilai mean dan standar deviasi dari tiga saluran RGB
# untuk menstandarisasi setiap saluran
normalize = gluon.data.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomResizedCrop(224),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    normalize])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    normalize])
```

```{.python .input}
#@tab pytorch
# Tentukan nilai mean dan standar deviasi dari tiga saluran RGB
# untuk menstandarisasi setiap saluran
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])
```

### [**Mendefinisikan dan Menginisialisasi Model**]

Kami menggunakan ResNet-18, yang telah dipra-latih pada dataset ImageNet, sebagai model sumber. Di sini, kami menentukan `pretrained=True` untuk secara otomatis mengunduh parameter model yang telah dipra-latih.  
Jika model ini digunakan untuk pertama kalinya,  
koneksi internet diperlukan untuk mengunduh.


```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
```

:begin_tab:`mxnet`
Instansi model sumber yang dipra-latih berisi dua variabel anggota: `features` dan `output`. Yang pertama berisi semua lapisan model kecuali lapisan output, dan yang terakhir adalah lapisan output dari model.  
Tujuan utama dari pembagian ini adalah untuk memfasilitasi fine-tuning parameter model pada semua lapisan kecuali lapisan output. Variabel anggota `output` dari model sumber ditunjukkan di bawah ini.
:end_tab:

:begin_tab:`pytorch`
Instansi model sumber yang dipra-latih berisi sejumlah lapisan fitur dan lapisan output `fc`.  
Tujuan utama dari pembagian ini adalah untuk memfasilitasi fine-tuning parameter model pada semua lapisan kecuali lapisan output. Variabel anggota `fc` dari model sumber diberikan di bawah ini.
:end_tab:


```{.python .input}
#@tab mxnet
pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net.fc
```

Sebagai lapisan terhubung sepenuhnya (fully connected), lapisan ini mengubah output global average pooling terakhir dari ResNet menjadi 1000 output kelas dari dataset ImageNet.  
Kemudian, kami membangun jaringan saraf baru sebagai model target. Model ini didefinisikan dengan cara yang sama seperti model sumber yang dipra-latih, kecuali bahwa  
jumlah output pada lapisan terakhirnya disesuaikan dengan jumlah kelas dalam dataset target (bukan 1000).

Pada kode di bawah ini, parameter model sebelum lapisan output dari instansi model target `finetune_net` diinisialisasi dengan parameter model dari lapisan yang sesuai pada model sumber.  
Karena parameter model ini diperoleh melalui pelatihan sebelumnya pada ImageNet,  
mereka efektif.  
Oleh karena itu, kita hanya dapat menggunakan  
laju pembelajaran kecil untuk *fine-tune* parameter yang dipra-latih tersebut.  
Sebaliknya, parameter model pada lapisan output diinisialisasi secara acak dan umumnya memerlukan laju pembelajaran yang lebih besar untuk dipelajari dari awal.  
Jika laju pembelajaran dasar adalah $\eta$, maka laju pembelajaran sebesar $10\eta$ akan digunakan untuk mengiterasi parameter model pada lapisan output.


```{.python .input}
#@tab mxnet
finetune_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
# Parameter model pada lapisan output akan diiterasi menggunakan laju pembelajaran yang sepuluh kali lebih besar
finetune_net.output.collect_params().setattr('lr_mult', 10)
```

```{.python .input}
#@tab pytorch
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight);
```

### [**Fine-Tuning Model**]

Pertama, kami mendefinisikan fungsi pelatihan `train_fine_tuning` yang menggunakan fine-tuning sehingga dapat dipanggil berkali-kali.


```{.python .input}
#@tab mxnet
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    devices = d2l.try_all_gpus()
    net.collect_params().reset_ctx(devices)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

```{.python .input}
#@tab pytorch
# If `param_group=True`, the model parameters in the output layer will be
# updated using a learning rate ten times greater
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

Kami [**menetapkan laju pembelajaran dasar ke nilai yang kecil**]  
untuk *fine-tune* parameter model yang diperoleh melalui pretraining. Berdasarkan pengaturan sebelumnya, kami akan melatih parameter lapisan output dari model target dari awal menggunakan laju pembelajaran yang sepuluh kali lebih besar.


```{.python .input}
#@tab mxnet
train_fine_tuning(finetune_net, 0.01)
```

```{.python .input}
#@tab pytorch
train_fine_tuning(finetune_net, 5e-5)
```

[**Untuk perbandingan,**] kami mendefinisikan model yang identik, tetapi (**menginisialisasi semua parameter modelnya dengan nilai acak**).
Karena seluruh model perlu dilatih dari awal, kami dapat menggunakan laju pembelajaran yang lebih besar.



```{.python .input}
#@tab mxnet
scratch_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1)
```

```{.python .input}
#@tab pytorch
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
```

Seperti yang kita lihat, model yang telah di-fine-tune cenderung tampil lebih baik untuk epoch yang sama  
karena nilai parameter awalnya lebih efektif.



## Ringkasan

* Transfer learning mentransfer pengetahuan yang dipelajari dari dataset sumber ke dataset target. Fine-tuning adalah teknik umum untuk transfer learning.
* Model target menyalin semua desain model beserta parameter-parameter mereka dari model sumber kecuali lapisan output, dan melakukan fine-tune pada parameter-parameter ini berdasarkan dataset target. Sebaliknya, lapisan output dari model target perlu dilatih dari awal.
* Secara umum, fine-tuning parameter menggunakan laju pembelajaran yang lebih kecil, sementara melatih lapisan output dari awal dapat menggunakan laju pembelajaran yang lebih besar.


## Latihan

1. Terus tingkatkan laju pembelajaran dari `finetune_net`. Bagaimana perubahan akurasi model?
2. Sesuaikan lagi hiperparameter dari `finetune_net` dan `scratch_net` dalam eksperimen perbandingan. Apakah mereka masih berbeda dalam akurasi?
3. Tentukan parameter sebelum lapisan output dari `finetune_net` sesuai dengan parameter dari model sumber dan *jangan* perbarui selama pelatihan. Bagaimana perubahan akurasi model? Anda dapat menggunakan kode berikut.


```{.python .input}
#@tab mxnet
finetune_net.features.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
for param in finetune_net.parameters():
    param.requires_grad = False
```

4. Sebenarnya, terdapat kelas "hotdog" dalam dataset `ImageNet`. Parameter bobot yang sesuai di lapisan output dapat diperoleh melalui kode berikut. Bagaimana kita dapat memanfaatkan parameter bobot ini?


```{.python .input}
#@tab mxnet
weight = pretrained_net.output.weight
hotdog_w = np.split(weight.data(), 1000, axis=0)[713]
hotdog_w.shape
```

```{.python .input}
#@tab pytorch
weight = pretrained_net.fc.weight
hotdog_w = torch.split(weight.data, 1, dim=0)[934]
hotdog_w.shape
```

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/368)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1439)
:end_tab:
