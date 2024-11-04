# Jaringan Neural Linear untuk Regresi
:label:`chap_regression`

Sebelum kita mulai memikirkan untuk membuat jaringan neural yang mendalam,
akan sangat membantu untuk mengimplementasikan jaringan yang dangkal terlebih dahulu,
di mana input terhubung langsung ke output.
Ini akan penting untuk beberapa alasan.
Pertama, alih-alih terganggu oleh arsitektur yang rumit,
kita bisa fokus pada dasar-dasar pelatihan jaringan neural,
termasuk parametrisasi lapisan output, menangani data,
menentukan fungsi kerugian, dan melatih model.
Kedua, kelas jaringan dangkal ini kebetulan mencakup
set dari model linear,
yang mencakup banyak metode prediksi statistik klasik,
termasuk regresi linear dan softmax.
Memahami alat-alat klasik ini sangat penting
karena mereka digunakan secara luas dalam banyak konteks
dan kita sering perlu menggunakannya sebagai baseline
saat membenarkan penggunaan arsitektur yang lebih canggih.
Bab ini akan fokus secara sempit pada regresi linear
dan bab berikutnya akan memperluas repertori pemodelan kita
dengan mengembangkan jaringan neural linear untuk klasifikasi.

```toc
:maxdepth: 2

linear-regression
oo-design
synthetic-regression-data
linear-regression-scratch
linear-regression-concise
generalization
weight-decay
```

