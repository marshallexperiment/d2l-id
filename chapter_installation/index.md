# Instalasi
:label:`chap_installation`

Untuk dapat memulai, kita memerlukan lingkungan untuk menjalankan Python, Jupyter Notebook, pustaka yang relevan, dan kode yang diperlukan untuk menjalankan buku ini sendiri.

## Menginstal Miniconda

Pilihan paling sederhana adalah menginstal [Miniconda](https://conda.io/en/latest/miniconda.html). Perlu dicatat bahwa versi Python 3.x diperlukan. Anda dapat melewati langkah-langkah berikut jika mesin Anda sudah memiliki conda terinstal.

Kunjungi situs Miniconda dan tentukan versi yang sesuai untuk sistem Anda berdasarkan versi Python 3.x dan arsitektur mesin Anda. Misalkan versi Python Anda adalah 3.9 (versi yang sudah kami uji). Jika Anda menggunakan macOS, unduh skrip bash yang namanya mengandung string "MacOSX", lalu navigasikan ke lokasi unduhan, dan jalankan instalasi sebagai berikut (mengambil contoh Intel Mac):

```bash
# Nama file dapat berubah
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```

Pengguna Linux akan mengunduh file yang namanya mengandung string "Linux" dan menjalankan perintah berikut di lokasi unduhan:

```bash
# Nama file dapat berubah
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```

Pengguna Windows akan mengunduh dan menginstal Miniconda dengan mengikuti [petunjuk online](https://conda.io/en/latest/miniconda.html). Di Windows, Anda dapat mencari `cmd` untuk membuka Command Prompt (interpretator baris perintah) untuk menjalankan perintah.

Selanjutnya, inisialisasi shell agar kita bisa menjalankan `conda` langsung.

```bash
~/miniconda3/bin/conda init
```

Kemudian tutup dan buka kembali shell Anda saat ini. Anda seharusnya dapat membuat lingkungan baru seperti berikut:

```bash
conda create --name d2l python=3.9 -y
```

Sekarang kita dapat mengaktifkan lingkungan `d2l`:

```bash
conda activate d2l
```

## Menginstal Framework Pembelajaran Mendalam dan Paket `d2l`

Sebelum menginstal framework pembelajaran mendalam apa pun, pastikan terlebih dahulu apakah Anda memiliki GPU yang sesuai pada mesin Anda (GPU yang digunakan untuk tampilan pada laptop standar tidak relevan untuk tujuan kita). Misalnya, jika komputer Anda memiliki GPU NVIDIA dan telah menginstal [CUDA](https://developer.nvidia.com/cuda-downloads), maka Anda sudah siap. Jika mesin Anda tidak memiliki GPU, tidak perlu khawatir dulu. CPU Anda menyediakan cukup tenaga untuk menyelesaikan beberapa bab pertama. Namun, Anda mungkin ingin mengakses GPU sebelum menjalankan model yang lebih besar.

:begin_tab:`mxnet`

Untuk menginstal versi MXNet yang mendukung GPU, kita perlu mengetahui versi CUDA yang telah terinstal. Anda dapat memeriksa ini dengan menjalankan `nvcc --version` atau `cat /usr/local/cuda/version.txt`. Misalkan Anda telah menginstal CUDA 11.2, maka jalankan perintah berikut:

```bash
# Untuk pengguna macOS dan Linux
pip install mxnet-cu112==1.9.1

# Untuk pengguna Windows
pip install mxnet-cu112==1.9.1 -f https://dist.mxnet.io/python
```

Anda dapat mengganti dua digit terakhir sesuai dengan versi CUDA Anda, misalnya `cu101` untuk CUDA 10.1 dan `cu90` untuk CUDA 9.0.

Jika mesin Anda tidak memiliki GPU NVIDIA atau CUDA, Anda dapat menginstal versi CPU seperti berikut:

```bash
pip install mxnet==1.9.1
```

:end_tab:

:begin_tab:`pytorch`

Anda dapat menginstal PyTorch (versi yang ditentukan sudah diuji pada saat penulisan ini) dengan dukungan CPU atau GPU sebagai berikut:

```bash
pip install torch==2.0.0 torchvision==0.15.1
```

:end_tab:

:begin_tab:`tensorflow`
Anda dapat menginstal TensorFlow dengan dukungan CPU atau GPU sebagai berikut:

```bash
pip install tensorflow==2.12.0 tensorflow-probability==0.20.0
```

:end_tab:

:begin_tab:`jax`
Anda dapat menginstal JAX dan Flax dengan dukungan CPU atau GPU sebagai berikut:

```bash
# GPU
pip install "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax==0.7.0
```

Jika mesin Anda tidak memiliki GPU NVIDIA atau CUDA, Anda dapat menginstal versi CPU sebagai berikut:

```bash
# CPU
pip install "jax[cpu]==0.4.13" flax==0.7.0
```

:end_tab:

Langkah selanjutnya adalah menginstal paket `d2l` yang kami kembangkan untuk mengenkapsulasi fungsi dan kelas yang sering digunakan di seluruh buku ini:

```bash
pip install d2l==1.0.3
```

## Mengunduh dan Menjalankan Kode

Selanjutnya, Anda perlu mengunduh notebook agar dapat menjalankan setiap blok kode dalam buku ini. Cukup klik tab "Notebooks" di bagian atas halaman HTML mana pun di [situs D2L.ai](https://d2l.ai/) untuk mengunduh kodenya dan kemudian unzip file tersebut. Sebagai alternatif, Anda dapat mengambil notebook dari baris perintah sebagai berikut:

:begin_tab:`mxnet`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd mxnet
```

:end_tab:

:begin_tab:`pytorch`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd pytorch
```

:end_tab:

:begin_tab:`tensorflow`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd tensorflow
```

:end_tab:

:begin_tab:`jax`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd jax
```

:end_tab:

Jika Anda belum memiliki `unzip` terinstal, jalankan terlebih dahulu `sudo apt-get install unzip`. Sekarang kita dapat memulai server Jupyter Notebook dengan menjalankan:

```bash
jupyter notebook
```

Pada titik ini, Anda dapat membuka http://localhost:8888 (mungkin sudah terbuka secara otomatis) di peramban web Anda. Setelah itu kita dapat menjalankan kode untuk setiap bagian dari buku ini. Setiap kali Anda membuka jendela baris perintah baru, Anda perlu menjalankan `conda activate d2l` untuk mengaktifkan lingkungan runtime sebelum menjalankan notebook D2L, atau memperbarui paket Anda (baik framework pembelajaran mendalam atau paket `d2l`). Untuk keluar dari lingkungan, jalankan `conda deactivate`.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/436)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17964)
:end_tab:
