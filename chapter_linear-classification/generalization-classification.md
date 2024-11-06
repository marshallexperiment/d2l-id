# Generalisasi dalam Klasifikasi

:label:`chap_classification_generalization`

Sejauh ini, kita telah berfokus pada cara mengatasi masalah klasifikasi multikelas 
dengan melatih jaringan saraf (linear) dengan output ganda dan fungsi softmax. 
Dengan menafsirkan keluaran model kita sebagai prediksi probabilistik, kita memotivasi dan menurunkan fungsi loss cross-entropy, 
yang menghitung kemungkinan log negatif yang ditetapkan oleh model kita 
(untuk set parameter yang tetap) pada label yang sebenarnya. 
Akhirnya, kita menerapkan alat ini dengan menyesuaikan model kita pada set pelatihan. Namun, seperti biasa, 
tujuan kita adalah mempelajari *pola umum*, yang dinilai secara empiris pada data yang belum pernah dilihat sebelumnya (set uji). 
Akurasi tinggi pada set pelatihan tidak berarti apa-apa.

Ketika setiap input kita unik (dan memang ini benar untuk sebagian besar dataset berdimensi tinggi), 
kita dapat mencapai akurasi sempurna pada set pelatihan dengan hanya menghafal dataset pada epoch pelatihan pertama
dan kemudian melihat label saat kita melihat gambar baru. Namun, menghafal label yang tepat yang terkait dengan contoh 
pelatihan yang tepat tidak memberi tahu kita bagaimana mengklasifikasikan contoh baru. 
Tanpa panduan lebih lanjut, kita mungkin harus mengandalkan tebakan acak setiap kali kita menemui contoh baru.

Beberapa pertanyaan penting yang membutuhkan perhatian segera:

1. Berapa banyak contoh uji yang kita butuhkan untuk memberikan perkiraan yang baik dari akurasi pengklasifikasi kita pada populasi dasar?
2. Apa yang terjadi jika kita terus mengevaluasi model pada set uji yang sama berulang kali?
3. Mengapa kita harus berharap bahwa menghubungkan model linear kita dengan set pelatihan harus berjalan lebih baik daripada skema hafalan naif kita?


Sementara :numref:`sec_generalization_basics` memperkenalkan dasar-dasar overfitting dan generalisasi dalam konteks regresi linear, 
bab ini akan membahas lebih dalam, memperkenalkan beberapa ide dasar dari teori pembelajaran statistik. 
Ternyata kita sering kali dapat menjamin generalisasi *a priori*: untuk banyak model, dan untuk batas atas yang diinginkan pada kesenjangan generalisasi $\epsilon$, 
kita sering kali dapat menentukan jumlah sampel yang diperlukan $n$ sedemikian rupa sehingga jika set pelatihan kita mengandung setidaknya $n$ sampel, 
error empiris kita akan berada dalam $\epsilon$ dari error sesungguhnya, *untuk setiap distribusi pembangkitan data*.

Sayangnya, juga ternyata bahwa meskipun jaminan semacam ini menyediakan dasar intelektual yang kuat,
mereka memiliki kegunaan praktis yang terbatas bagi praktisi deep learning. 
Singkatnya, jaminan ini menunjukkan bahwa memastikan generalisasi dari 
jaringan saraf dalam *a priori* memerlukan jumlah contoh yang luar biasa banyak (mungkin triliunan atau lebih), 
bahkan ketika kita menemukan bahwa, pada tugas-tugas yang kita pedulikan, 
jaringan saraf dalam umumnya dapat melakukan generalisasi dengan sangat baik dengan jumlah contoh yang jauh lebih sedikit (ribuan).

Oleh karena itu, praktisi deep learning sering kali mengesampingkan jaminan *a priori* sepenuhnya 
dan sebagai gantinya menggunakan metode yang telah terbukti melakukan generalisasi dengan baik pada masalah serupa di masa lalu, serta mengesahkan generalisasi *secara post hoc* melalui evaluasi empiris. 
Ketika kita sampai pada :numref:`chap_perceptrons`, 
kita akan kembali membahas generalisasi dan memberikan pengantar ringan pada literatur ilmiah yang luas yang telah berkembang dalam upaya untuk 
menjelaskan mengapa jaringan saraf dalam dapat melakukan generalisasi dalam praktik.


## Test Set

Karena kita telah mulai mengandalkan set uji sebagai metode standar emas untuk menilai error generalisasi, mari kita mulai dengan membahas sifat-sifat dari perkiraan error tersebut. 
Mari kita fokus pada pengklasifikasi tetap $f$, tanpa memusingkan bagaimana pengklasifikasi tersebut diperoleh. 
Selain itu, anggaplah bahwa kita memiliki sebuah *dataset baru* dari contoh $\mathcal{D} = {(\mathbf{x}^{(i)},y^{(i)})}_{i=1}^n$ yang tidak digunakan untuk melatih pengklasifikasi $f$. 
*Error empiris* dari pengklasifikasi kita $f$ pada $\mathcal{D}$ adalah fraksi dari contoh yang prediksinya $f(\mathbf{x}^{(i)})$ tidak sesuai dengan label sebenarnya $y^{(i)}$, yang diberikan oleh rumus berikut:

$$\epsilon_\mathcal{D}(f) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(f(\mathbf{x}^{(i)}) \neq y^{(i)}).$$

Sebaliknya, *error populasi* adalah fraksi *yang diharapkan* dari contoh dalam populasi dasar (suatu distribusi $P(X,Y)$ yang ditandai dengan fungsi kepadatan probabilitas $p(\mathbf{x},y)$) di mana pengklasifikasi 
kita tidak sesuai dengan label yang sebenarnya:

$$\epsilon(f) =  E_{(\mathbf{x}, y) \sim P} \mathbf{1}(f(\mathbf{x}) \neq y) =
\int\int \mathbf{1}(f(\mathbf{x}) \neq y) p(\mathbf{x}, y) \;d\mathbf{x} dy.$$

Meskipun $\epsilon(f)$ adalah besaran yang benar-benar kita perhatikan, kita tidak dapat mengamatinya secara langsung, 
sama halnya dengan kita tidak dapat mengamati tinggi rata-rata dalam populasi besar tanpa mengukur setiap orang. 
Kita hanya bisa memperkirakan besaran ini berdasarkan sampel. Karena set uji kita $\mathcal{D}$ secara statistik representatif terhadap populasi dasar, 
kita dapat melihat $\epsilon_\mathcal{D}(f)$ sebagai estimator statistik dari error populasi $\epsilon(f)$. 
Selain itu, karena besaran yang kita minati $\epsilon(f)$ adalah ekspektasi (dari variabel acak $\mathbf{1}(f(X) \neq Y)$) 
dan estimator yang sesuai $\epsilon_\mathcal{D}(f)$ adalah rata-rata sampel, 
memperkirakan error populasi hanyalah masalah klasik dalam estimasi rata-rata, yang mungkin Anda ingat dari :numref:`sec_prob`.


Hasil penting dari teori probabilitas klasik yang disebut *teorema limit pusat* menjamin bahwa ketika kita memiliki $n$ sampel acak $a_1, ..., a_n$ 
yang diambil dari distribusi apapun dengan rata-rata $\mu$ dan standar deviasi $\sigma$, maka ketika jumlah sampel $n$ mendekati tak hingga, 
rata-rata sampel $\hat{\mu}$ akan mendekati distribusi normal yang terpusat pada rata-rata sebenarnya dan dengan standar deviasi $\sigma/\sqrt{n}$. 
Ini memberikan kita suatu wawasan penting: ketika jumlah contoh bertambah banyak, error pengujian $\epsilon_\mathcal{D}(f)$ kita akan 
mendekati error sebenarnya $\epsilon(f)$ dengan laju $\mathcal{O}(1/\sqrt{n})$. 

Dengan kata lain, untuk memperkirakan error pengujian kita dengan dua kali lebih presisi, 
kita harus mengumpulkan set pengujian yang empat kali lebih besar. 
Untuk mengurangi error pengujian kita hingga seratus kali lebih kecil, kita harus mengumpulkan set pengujian sepuluh ribu kali lebih besar. 
Secara umum, laju $\mathcal{O}(1/\sqrt{n})$ ini sering kali menjadi harapan terbaik yang dapat kita capai dalam statistik.

Sekarang, setelah kita mengetahui sesuatu tentang laju asimptotik di mana error pengujian $\epsilon_\mathcal{D}(f)$ kita mendekati error 
sebenarnya $\epsilon(f)$, kita dapat menyoroti beberapa detail penting. Ingat bahwa variabel acak yang kita minati $\mathbf{1}(f(X) \neq Y)$ 
hanya dapat bernilai $0$ dan $1$, sehingga merupakan variabel acak Bernoulli, 
yang ditandai oleh suatu parameter yang menunjukkan probabilitas bahwa ia bernilai $1$. Di sini, $1$ berarti 
bahwa pengklasifikasi kita membuat kesalahan, sehingga parameter dari variabel acak kita sebenarnya adalah tingkat error sebenarnya $\epsilon(f)$. 

Variansi $\sigma^2$ dari variabel Bernoulli tergantung pada parameternya (di sini, $\epsilon(f)$) 
menurut persamaan $\epsilon(f)(1-\epsilon(f))$. Walaupun $\epsilon(f)$ pada awalnya tidak diketahui, 
kita tahu bahwa nilainya tidak mungkin lebih besar dari $1$. Sedikit investigasi terhadap fungsi ini 
menunjukkan bahwa variansi kita adalah yang tertinggi ketika tingkat error sebenarnya mendekati $0.5$ dan 
dapat jauh lebih rendah ketika mendekati $0$ atau $1$. Ini memberi tahu kita bahwa standar deviasi asimptotik 
dari perkiraan kita $\epsilon_\mathcal{D}(f)$ terhadap error $\epsilon(f)$ (atas pilihan dari $n$ sampel uji) tidak bisa lebih besar dari $\sqrt{0.25/n}$.


Jika kita mengabaikan fakta bahwa laju ini menggambarkan perilaku ketika 
ukuran set pengujian mendekati tak hingga daripada ketika kita memiliki sampel yang terbatas, 
ini menunjukkan bahwa jika kita ingin error pengujian $\epsilon_\mathcal{D}(f)$ 
mendekati error populasi $\epsilon(f)$ sehingga satu standar deviasi 
sesuai dengan interval $\pm 0.01$, maka kita harus mengumpulkan sekitar 2500 sampel. 
Jika kita ingin memasukkan dua standar deviasi dalam rentang tersebut 
dan dengan demikian memiliki keyakinan 95% bahwa $\epsilon_\mathcal{D}(f) \in \epsilon(f) \pm 0.01$, 
maka kita akan memerlukan 10.000 sampel!

Ukuran ini ternyata sesuai dengan ukuran set pengujian pada banyak tolok ukur (benchmark) populer dalam machine learning. 
Anda mungkin akan terkejut mengetahui bahwa ribuan makalah deep learning terapan diterbitkan setiap tahun, 
yang menyoroti peningkatan error rate sebesar $0.01$ atau kurang. Tentu saja, ketika tingkat error mendekati $0$, 
maka peningkatan sebesar $0.01$ memang bisa sangat berarti.

Salah satu karakteristik yang agak mengganggu dari analisis kita sejauh ini adalah 
bahwa analisis tersebut hanya benar-benar memberikan 
informasi mengenai asimptotik, yaitu, bagaimana hubungan antara $\epsilon_\mathcal{D}$ dan $\epsilon$ 
berkembang ketika ukuran sampel menuju tak hingga. 
Untungnya, karena variabel acak kita terbatas, kita dapat memperoleh batasan sampel yang valid 
dengan menggunakan sebuah ketidaksamaan yang diperkenalkan oleh Hoeffding (1963):

$$P(\epsilon_\mathcal{D}(f) - \epsilon(f) \geq t) < \exp\left( - 2n t^2 \right).$$

Dengan menyelesaikan untuk ukuran dataset terkecil yang memungkinkan kita menyimpulkan dengan keyakinan 95% bahwa jarak $t$ 
antara perkiraan kita $\epsilon_\mathcal{D}(f)$ dan tingkat error sebenarnya $\epsilon(f)$ tidak melebihi $0.01$, 
Anda akan menemukan bahwa diperlukan sekitar 15.000 contoh, 
dibandingkan dengan 10.000 contoh yang disarankan oleh analisis asimptotik di atas. 
Jika Anda mendalami statistik lebih jauh, Anda akan menemukan bahwa tren ini berlaku secara umum. 
Jaminan yang tetap berlaku bahkan dalam sampel terbatas biasanya sedikit lebih konservatif. 
Perlu dicatat bahwa dalam skema besar, angka-angka ini tidak terlalu jauh berbeda, 
yang mencerminkan kegunaan umum dari analisis asimptotik untuk memberikan kita perkiraan kasar 
meskipun bukan jaminan yang dapat kita jadikan pegangan pasti.


## Penggunaan Ulang Test Set

Dalam beberapa hal, Anda sekarang siap untuk berhasil dalam melakukan penelitian empiris dalam machine learning. 
Hampir semua model praktis dikembangkan dan divalidasi berdasarkan kinerja test set, 
dan Anda kini telah menguasai penggunaan test set. Untuk setiap classifier tetap $f$, 
Anda tahu cara mengevaluasi error pengujiannya $\epsilon_\mathcal{D}(f)$, 
dan tahu dengan tepat apa yang bisa (dan tidak bisa) dikatakan tentang error populasinya $\epsilon(f)$.

Jadi, katakanlah Anda menggunakan pengetahuan ini dan bersiap untuk melatih model pertama Anda $f_1$. 
Mengetahui seberapa yakin Anda perlu dalam tingkat error classifier Anda, 
Anda menerapkan analisis di atas untuk menentukan jumlah contoh yang sesuai untuk dialokasikan ke test set. 
Selain itu, misalkan Anda telah mempelajari dengan seksama pelajaran dari :numref:`sec_generalization_basics` 
dan memastikan untuk menjaga kemurnian test set dengan melakukan semua analisis awal, tuning hyperparameter, 
dan bahkan pemilihan di antara beberapa arsitektur model yang bersaing di validation set.

Akhirnya, Anda mengevaluasi model $f_1$ Anda pada test set 
dan melaporkan estimasi yang tidak bias dari error populasi 
dengan interval kepercayaan yang terkait. 
Hingga saat ini, semuanya tampak berjalan dengan baik. 
Namun, malam itu, Anda terbangun pukul 3 pagi 
dengan ide brilian untuk pendekatan pemodelan baru.

Keesokan harinya, Anda menulis kode untuk model baru Anda, men-tune hyperparameternya pada validation set, 
dan ternyata model baru $f_2$ Anda tidak hanya berfungsi, tetapi juga menunjukkan error rate yang tampaknya jauh lebih rendah dibandingkan $f_1$. 
Namun, kegembiraan atas penemuan ini tiba-tiba memudar saat Anda bersiap untuk evaluasi akhir. Anda tidak memiliki test set!


Meskipun test set asli $\mathcal{D}$ masih berada di server Anda, kini Anda menghadapi dua masalah besar. 
Pertama, saat Anda mengumpulkan test set, Anda menentukan tingkat presisi yang diperlukan dengan asumsi 
bahwa Anda hanya mengevaluasi satu classifier $f$. Namun, jika Anda mulai mengevaluasi beberapa classifier $f_1, ..., f_k$ pada test set yang sama, 
Anda harus mempertimbangkan masalah *false discovery*. Sebelumnya, Anda mungkin 95% yakin bahwa $\epsilon_\mathcal{D}(f) \in \epsilon(f) \pm 0.01$ 
untuk satu classifier $f$, sehingga kemungkinan hasil yang menyesatkan hanya 5%. Dengan $k$ classifier dalam campuran, sulit menjamin bahwa 
tidak ada satu pun di antara mereka yang kinerjanya pada test set ternyata menyesatkan. Dengan 20 classifier yang dipertimbangkan, 
Anda mungkin sama sekali tidak memiliki kekuatan untuk menyingkirkan kemungkinan bahwa setidaknya satu dari mereka mendapatkan skor yang menyesatkan. 
Masalah ini terkait dengan *multiple hypothesis testing*, 
yang meskipun memiliki literatur yang luas dalam statistik, tetap menjadi masalah yang mengganggu penelitian ilmiah.

Jika itu belum cukup membuat Anda khawatir, ada alasan khusus untuk meragukan hasil yang Anda peroleh pada evaluasi selanjutnya. 
Ingat bahwa analisis kinerja test set kita didasarkan pada asumsi bahwa classifier dipilih tanpa kontak dengan test set sehingga
kita dapat menganggap test set diambil secara acak dari populasi dasar. Di sini, bukan hanya Anda menguji beberapa fungsi, 
tetapi fungsi selanjutnya $f_2$ dipilih setelah Anda mengamati kinerja test set dari $f_1$. Begitu informasi dari test set 
bocor ke pihak yang membuat model, test set tersebut tidak pernah bisa menjadi test set sejati lagi dalam pengertian yang paling ketat. 
Masalah ini disebut *adaptive overfitting* dan baru-baru ini menjadi 
topik yang menarik perhatian para teoritikus pembelajaran dan ahli statistik :cite:`dwork2015preserving`.

Untungnya, meskipun mungkin untuk membocorkan semua informasi dari holdout set, 
dan skenario terburuk secara teori tampak suram, analisis ini mungkin terlalu konservatif. 
Dalam praktiknya, berhati-hatilah untuk membuat test set yang sesungguhnya, untuk merujuknya sesedikit mungkin, 
untuk mempertimbangkan *multiple hypothesis testing* saat melaporkan interval kepercayaan, dan untuk meningkatkan 
kewaspadaan Anda lebih agresif ketika risikonya tinggi dan ukuran dataset Anda kecil. 
Saat menjalankan serangkaian tantangan benchmark, 
seringkali merupakan praktik yang baik untuk mempertahankan beberapa test set sehingga setelah setiap putaran, 
test set lama dapat diturunkan menjadi validation set.


## Teori Pembelajaran Statistik

Sederhananya, *test set adalah satu-satunya yang kita miliki*, 
namun fakta ini terasa kurang memuaskan. Pertama, jarang sekali kita 
memiliki *test set yang sejati*—kecuali jika kita sendiri yang membuat dataset, 
kemungkinan orang lain sudah mengevaluasi classifier mereka sendiri pada "test set" kita. 
Dan bahkan ketika kita yang pertama kali menggunakannya, kita akan segera merasa frustrasi, 
ingin bisa mengevaluasi upaya pemodelan kita berikutnya tanpa merasa tidak yakin apakah kita bisa mempercayai hasilnya. 
Selain itu, bahkan test set sejati hanya dapat memberi tahu kita *post hoc* apakah suatu 
classifier benar-benar dapat melakukan generalisasi pada populasi, bukan apakah kita punya alasan untuk berharap *a priori* bahwa itu seharusnya bisa melakukan generalisasi.

Dengan keraguan ini, Anda mungkin sudah cukup siap untuk melihat daya tarik dari *teori pembelajaran statistik*, 
subbidang matematika dalam machine learning yang berfokus pada prinsip-prinsip mendasar yang menjelaskan mengapa/kapan model 
yang dilatih pada data empiris dapat/akan melakukan generalisasi terhadap data yang belum terlihat. 
Salah satu tujuan utama para peneliti pembelajaran statistik adalah untuk membatasi celah generalisasi, 
yang menghubungkan sifat-sifat kelas model dengan jumlah sampel dalam dataset.

Para teoritikus pembelajaran berupaya membatasi perbedaan antara *empirical error* $\epsilon_\mathcal{S}(f_\mathcal{S})$ dari 
classifier yang dipelajari $f_\mathcal{S}$, yang dilatih dan dievaluasi pada training set $\mathcal{S}$, dengan error sebenarnya $\epsilon(f_\mathcal{S})$ dari 
classifier yang sama pada populasi dasar. Ini mungkin terlihat mirip dengan masalah evaluasi yang baru saja kita bahas, 
tetapi ada perbedaan besar. Sebelumnya, classifier $f$ bersifat tetap dan kita hanya memerlukan dataset untuk tujuan evaluasi.
Dan memang, classifier yang tetap dapat melakukan generalisasi: error-nya pada dataset (yang belum pernah terlihat sebelumnya) adalah perkiraan yang tidak bias dari error populasi.

Namun, apa yang bisa kita katakan ketika classifier dilatih dan dievaluasi pada dataset yang sama? 
Bisakah kita yakin bahwa error pelatihan akan mendekati error pengujian?



Misalkan classifier yang kita pelajari $f_\mathcal{S}$ harus dipilih dari 
suatu himpunan fungsi yang telah ditentukan sebelumnya, yaitu $\mathcal{F}$. 
Ingat dari diskusi kita tentang test set bahwa walaupun mudah untuk memperkirakan error dari satu classifier, 
situasinya menjadi rumit ketika kita mulai mempertimbangkan sekumpulan classifier. Meskipun *empirical error* 
dari setiap satu classifier (tetap) cenderung mendekati error sebenarnya dengan probabilitas tinggi, 
begitu kita mempertimbangkan sekumpulan classifier, 
kita perlu khawatir tentang kemungkinan bahwa *hanya satu* dari mereka memiliki estimasi error yang buruk.

Kekhawatirannya adalah bahwa kita mungkin memilih classifier 
seperti itu dan dengan demikian secara drastis meremehkan error populasi. 
Selain itu, bahkan untuk model linear, karena parameter-parameter dari model tersebut 
memiliki nilai yang kontinu, kita biasanya memilih dari 
kelas fungsi yang tidak terbatas ($|\mathcal{F}| = \infty$).


Salah satu solusi ambisius untuk masalah ini adalah mengembangkan alat analitik untuk membuktikan *uniform convergence*, 
yaitu bahwa dengan probabilitas tinggi, *empirical error rate* untuk setiap classifier dalam kelas $f \in \mathcal{F}$ 
akan *secara simultan* konvergen ke *true error rate* mereka masing-masing. Dengan kata lain, kita mencari prinsip 
teoretis yang memungkinkan kita menyatakan bahwa dengan probabilitas setidaknya $1 - \delta$ (untuk beberapa nilai kecil $\delta$), 
tidak ada tingkat error classifier $\epsilon(f)$ (di antara semua classifier dalam kelas $\mathcal{F}$) 
yang salah diperkirakan lebih dari jumlah kecil $\alpha$. Jelas, kita tidak bisa membuat pernyataan seperti itu untuk semua kelas model $\mathcal{F}$. 
Ingat kelas *memorization machines* yang selalu mencapai error empiris $0$ tetapi tidak pernah melebihi tebakan acak pada populasi yang mendasarinya.

Dalam beberapa hal, kelas memorizers terlalu fleksibel. 
Tidak ada hasil *uniform convergence* yang mungkin berlaku. 
Di sisi lain, classifier tetap (fixed classifier) tidak berguna—ia menggeneralisasi dengan sempurna, 
tetapi tidak sesuai dengan data pelatihan atau data pengujian. Pertanyaan utama dalam pembelajaran historisnya dirumuskan sebagai 
keseimbangan antara kelas model yang lebih fleksibel (dengan varians tinggi) yang lebih sesuai dengan data pelatihan tetapi berisiko overfitting, 
versus kelas model yang lebih kaku (dengan bias tinggi) yang dapat melakukan generalisasi dengan baik tetapi berisiko underfitting.
Pertanyaan inti dalam teori pembelajaran adalah mengembangkan analisis matematika yang sesuai untuk mengukur di mana suatu model berada dalam spektrum ini, dan memberikan jaminan yang terkait.

Dalam serangkaian makalah yang berpengaruh, Vapnik dan Chervonenkis memperluas teori tentang konvergensi 
frekuensi relatif untuk kelas fungsi yang lebih umum :cite:`VapChe64,VapChe68,VapChe71,VapChe74b,VapChe81,VapChe91`.
Salah satu kontribusi utama dari 
penelitian ini adalah *Vapnik--Chervonenkis (VC) dimension*, 
yang mengukur (salah satu konsep) kompleksitas (fleksibilitas) dari kelas model. 
Selain itu, salah satu hasil utama mereka mengikat perbedaan antara error empiris
dan error populasi sebagai fungsi dari VC dimension dan jumlah sampel:

$$P\left(R[p, f] - R_\textrm{emp}[\mathbf{X}, \mathbf{Y}, f] < \alpha\right) \geq 1-\delta \ \textrm{ untuk }\ \alpha \geq c \sqrt{(\textrm{VC} - \log \delta)/n}.$$

Di sini, $\delta > 0$ adalah probabilitas bahwa batas tersebut dilanggar, $\alpha$ adalah batas 
atas pada *generalization gap*, dan $n$ adalah ukuran dataset. Terakhir, $c > 0$ adalah konstanta 
yang hanya bergantung pada skala kerugian yang dapat ditimbulkan. Salah satu penggunaan batas ini 
adalah dengan memasukkan nilai yang diinginkan dari $\delta$ dan $\alpha$ untuk menentukan berapa banyak 
sampel yang harus dikumpulkan. VC dimension mengukur jumlah titik data terbesar di mana kita dapat menetapkan pelabelan (biner) 
sembarang dan untuk masing-masing menemukan model $f$ dalam kelas yang sesuai dengan pelabelan tersebut. 
Sebagai contoh, model linear pada masukan berdimensi $d$ memiliki VC dimension $d+1$.
Mudah dilihat bahwa sebuah garis dapat menetapkan pelabelan apapun pada tiga titik dalam dua dimensi, tetapi tidak pada empat titik.

Sayangnya, teori ini cenderung terlalu pesimistis untuk model yang lebih kompleks, 
dan untuk mendapatkan jaminan ini biasanya membutuhkan lebih banyak contoh daripada yang sebenarnya 
diperlukan untuk mencapai tingkat error yang diinginkan. 
Perhatikan juga bahwa dengan memperbaiki kelas model dan $\delta$, 
tingkat error kita sekali lagi menurun dengan laju $\mathcal{O}(1/\sqrt{n})$ yang biasa. 
Tampaknya tidak mungkin kita bisa mendapatkan hasil yang lebih baik dalam hal $n$.
Namun, saat kita memvariasikan kelas model, VC dimension dapat menunjukkan gambaran pesimistis tentang *generalization gap*.


## Ringkasan

Cara paling sederhana untuk mengevaluasi sebuah model adalah dengan menggunakan test set 
yang terdiri dari data yang sebelumnya tidak terlihat. 
Evaluasi test set memberikan estimasi yang tidak bias dari error sebenarnya 
dan berkonvergen pada laju $\mathcal{O}(1/\sqrt{n})$ saat ukuran test set bertambah. 
Kita dapat menyediakan interval kepercayaan perkiraan berdasarkan distribusi asimptotik yang tepat 
atau interval kepercayaan sampel terbatas yang valid berdasarkan jaminan sampel terbatas (yang lebih konservatif). 
Evaluasi test set adalah landasan dari penelitian machine learning modern. Namun, test set jarang menjadi test set 
yang sejati (karena sering digunakan oleh banyak peneliti berulang kali). 
Ketika test set yang sama digunakan untuk mengevaluasi beberapa model, mengendalikan false discovery menjadi sulit, 
yang dapat menyebabkan masalah besar dalam teori. Dalam praktiknya, signifikansi masalah ini bergantung pada ukuran holdout 
set yang digunakan dan apakah set ini hanya digunakan untuk memilih hyperparameter atau jika informasi bocor lebih langsung.
Meskipun demikian, praktik yang baik adalah mengkurasi test set yang benar (atau beberapa set) 
dan bersikap se-konservatif mungkin tentang seberapa sering mereka digunakan.

Berharap menyediakan solusi yang lebih memuaskan, para teoritikus pembelajaran statistik telah mengembangkan metode untuk menjamin konvergensi seragam di seluruh kelas model. 
Jika memang error empiris setiap model secara simultan berkonvergen ke error sebenarnya, 
maka kita bebas memilih model yang berkinerja terbaik, meminimalkan error pelatihan, dengan mengetahui bahwa model tersebut juga akan berkinerja serupa pada data holdout. 
Secara krusial, setiap hasil semacam itu harus bergantung pada beberapa properti dari kelas model. 
Vladimir Vapnik dan Alexey Chervonenkis memperkenalkan *VC dimension*, 
yang menyajikan hasil konvergensi seragam yang berlaku untuk semua model dalam kelas VC. 
Error pelatihan untuk semua model dalam kelas ini (secara simultan) dijamin mendekati error sebenarnya, 
dan dijamin semakin mendekat pada laju $\mathcal{O}(1/\sqrt{n})$. Setelah penemuan revolusioner dari VC dimension, 
banyak ukuran kompleksitas alternatif telah diusulkan, masing-masing memfasilitasi jaminan generalisasi yang serupa. 
Lihat :citet:`boucheron2005theory` untuk pembahasan mendalam tentang berbagai cara lanjutan untuk mengukur kompleksitas fungsi.

Sayangnya, meskipun ukuran kompleksitas ini telah menjadi alat yang berguna dalam teori statistik, 
mereka ternyata tidak memiliki kekuatan (jika diterapkan secara langsung) untuk menjelaskan mengapa deep neural network mampu melakukan generalisasi. 
Deep neural network sering kali memiliki jutaan parameter (atau lebih), dan dengan mudah dapat menetapkan label acak pada banyak titik data. 
Namun demikian, mereka mampu melakukan generalisasi dengan baik pada masalah praktis dan, secara mengejutkan, 
seringkali melakukan generalisasi lebih baik saat modelnya lebih besar dan lebih dalam, meskipun memiliki VC dimension yang lebih tinggi.
Pada bab berikutnya, kita akan meninjau kembali generalisasi dalam konteks deep learning.

## Latihan

1. Jika kita ingin memperkirakan error dari model tetap $f$ dengan ketelitian hingga $0.0001$ dengan probabilitas lebih dari 99,9%, berapa banyak sampel yang kita butuhkan?
2. Misalkan seseorang memiliki test set berlabel $\mathcal{D}$ dan hanya menyediakan input tanpa label (fitur). Sekarang, anggap bahwa Anda hanya dapat mengakses label test set dengan menjalankan model $f$ (tanpa batasan pada kelas model) pada masing-masing input tanpa label dan menerima error yang sesuai $\epsilon_\mathcal{D}(f)$. Berapa banyak model yang perlu Anda evaluasi sebelum seluruh test set bocor dan Anda dapat terlihat memiliki error $0$, tanpa memandang error sebenarnya Anda?
3. Berapakah VC dimension dari kelas polinomial orde lima?
4. Berapakah VC dimension dari persegi panjang yang berorientasi sumbu pada data berdimensi dua?

[Diskusi](https://discuss.d2l.ai/t/6829)
