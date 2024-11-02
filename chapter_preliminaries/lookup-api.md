```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```
# Dokumentasi
:begin_tab:`mxnet`
Meskipun tidak mungkin bagi kami untuk memperkenalkan setiap fungsi dan kelas MXNet 
(dan informasinya mungkin cepat usang), 
[dokumentasi API](https://mxnet.apache.org/versions/1.8.0/api) 
dan [tutorial tambahan](https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/) serta contoh 
menyediakan dokumentasi semacam itu. 
Bagian ini memberikan panduan tentang cara menjelajahi API MXNet.
:end_tab:

:begin_tab:`pytorch`
Meskipun tidak mungkin bagi kami untuk memperkenalkan setiap fungsi dan kelas PyTorch 
(dan informasinya mungkin cepat usang), 
[dokumentasi API](https://pytorch.org/docs/stable/index.html) dan [tutorial tambahan](https://pytorch.org/tutorials/beginner/basics/intro.html) serta contoh 
menyediakan dokumentasi semacam itu.
Bagian ini memberikan panduan tentang cara menjelajahi API PyTorch.
:end_tab:

:begin_tab:`tensorflow`
Meskipun tidak mungkin bagi kami untuk memperkenalkan setiap fungsi dan kelas TensorFlow 
(dan informasinya mungkin cepat usang), 
[dokumentasi API](https://www.tensorflow.org/api_docs) dan [tutorial tambahan](https://www.tensorflow.org/tutorials) serta contoh 
menyediakan dokumentasi semacam itu. 
Bagian ini memberikan panduan tentang cara menjelajahi API TensorFlow.
:end_tab:



```{.python .input}
%%tab mxnet
from mxnet import np
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
import jax
```

## Fungsi dan Kelas dalam Modul

Untuk mengetahui fungsi dan kelas apa saja yang dapat dipanggil dalam sebuah modul,
kita memanggil fungsi `dir`. Misalnya, kita dapat
(**menanyakan semua properti dalam modul untuk menghasilkan bilangan acak**):


```{.python .input  n=1}
%%tab mxnet
print(dir(np.random))
```

```{.python .input  n=1}
%%tab pytorch
print(dir(torch.distributions))
```

```{.python .input  n=1}
%%tab tensorflow
print(dir(tf.random))
```

```{.python .input}
%%tab jax
print(dir(jax.random))
```

Secara umum, kita dapat mengabaikan fungsi-fungsi yang diawali dan diakhiri dengan `__` (objek khusus dalam Python) 
atau fungsi yang dimulai dengan satu `_` (biasanya fungsi internal). 
Berdasarkan nama fungsi atau atribut yang tersisa, 
kita bisa menebak bahwa modul ini menawarkan 
berbagai metode untuk menghasilkan bilangan acak, 
termasuk pengambilan sampel dari distribusi uniform (`uniform`), 
distribusi normal (`normal`), dan distribusi multinomial (`multinomial`).

## Fungsi dan Kelas Khusus

Untuk instruksi khusus tentang cara menggunakan suatu fungsi atau kelas tertentu,
kita dapat memanggil fungsi `help`. Sebagai contoh, mari
[**mengeksplorasi instruksi penggunaan untuk fungsi `ones` pada tensor**].


```{.python .input}
%%tab mxnet
help(np.ones)
```

```{.python .input}
%%tab pytorch
help(torch.ones)
```

```{.python .input}
%%tab tensorflow
help(tf.ones)
```

```{.python .input}
%%tab jax
help(jax.numpy.ones)
```

Dari dokumentasi, kita dapat melihat bahwa fungsi `ones` 
membuat tensor baru dengan bentuk yang ditentukan 
dan mengatur semua elemennya ke nilai 1. 
Sebisa mungkin, Anda sebaiknya (**menjalankan uji cepat**) 
untuk mengonfirmasi interpretasi Anda:


```{.python .input}
%%tab mxnet
np.ones(4)
```

```{.python .input}
%%tab pytorch
torch.ones(4)
```

```{.python .input}
%%tab tensorflow
tf.ones(4)
```

```{.python .input}
%%tab jax
jax.numpy.ones(4)
```

Di dalam Jupyter notebook, kita bisa menggunakan `?` untuk menampilkan dokumentasi di jendela lain. 
Misalnya, `list?` akan menampilkan konten
yang hampir identik dengan `help(list)`,
dalam jendela browser baru.
Selain itu, jika kita menggunakan dua tanda tanya, seperti `list??`,
kode Python yang mengimplementasikan fungsi tersebut juga akan ditampilkan.

Dokumentasi resmi menyediakan banyak deskripsi dan contoh yang lebih mendalam daripada buku ini. 
Kami menekankan kasus penggunaan penting 
yang akan membantu Anda memulai dengan cepat dalam menyelesaikan masalah praktis, 
bukan memberikan cakupan secara menyeluruh. 
Kami juga mendorong Anda untuk mempelajari kode sumber dari pustaka-pustaka ini 
untuk melihat contoh implementasi berkualitas tinggi dari kode produksi. 
Dengan melakukan ini, Anda tidak hanya akan menjadi insinyur yang lebih baik, 
tetapi juga ilmuwan yang lebih baik.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/199)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17972)
:end_tab:

