# Havalimanı Gürültü Tespit ve Ses Tanıma Sistemi

Havalimanı çevresindeki sesleri üst kategori ve alt tür düzeyinde inceleyen,
çoklu model kanıtını Shazam tarzı akustik parmak iziyle birleştiren masaüstü
uygulamasıdır. Proje; dosyadan analiz, çoklu pencere oylaması, uçak güvenlik
kapısı, insan onaylı referans üretimi ve salt-okunur SQLite kanıt ekranı sunar.

> Kaynak kod normal Git deposunda tutulur. Büyük model ağırlıkları, ses veri
> setleri ve SQLite indeksleri sürümlü GitHub **Release ZIP paketleri** olarak
> yayımlanır. İndirme ve geri yükleme adımları aşağıdadır. Yalnız paylaşım
> lisansı doğrulanmış veri paketleri herkese açık yayımlanmalıdır.

## Sistem ne yapar?

Ana sınıflandırıcı aşağıdaki altı üst kategoriden birini üretir:

| Etiket | Anlamı |
|---|---|
| `AIRCRAFT` | Uçak, motor, kalkış ve iniş sesleri |
| `AMBIENT` | Genel çevre ve havalimanı ortamı |
| `OTHER` | Tanımlı ana kategorilerin dışında kalan sesler |
| `SPEECH` | Konuşma ve anons |
| `TRAFFIC` | Kara ulaşımı ve trafik |
| `WIND` | Rüzgâr ve mikrofondaki hava akışı |

Üst kategori bulunduktan sonra ikinci aşama çalışır:

- `AIRCRAFT`: önce uçak parmak izi indeksinde kesin kayıt eşleşmesi aranır;
  eşleşme yoksa mevcut öğrenilmiş uçak-alt-tür modeli öneri üretir.
- `TRAFFIC`: bisiklet, otobüs, otomobil, motosiklet, tren ve kamyon başlığı
  çalışır.
- `OTHER`: kedi, karga, köpek, papağan, tavus kuşu ve serçe başlığı çalışır.
- Güvenilir alt tür modeli olmayan kategorilerde zorunlu tahmin yapılmaz.

## Mimari

```text
WAV / mikrofon
      |
      v
5 saniyelik pencereler (2,5 saniye adım, en fazla 5 temsilci pencere)
      |
      +--> EfficientNet-B0 / CNN / SVM / isteğe bağlı BEATs
      |             |
      |             v
      |      üst kategori + güven değerleri
      |             |
      |       Aircraft Guard
      |             |
      +-------------+
                    v
           çoklu pencere oylaması
                    |
                    v
        Shazam parmak izi (varsa kesin eşleşme)
                    |
                    +--> eşleşme yok: öğrenilmiş alt-tür modeli
                    |
                    v
          sonuç, yöntem ve insan denetim kanıtı
```

Shazam katmanı bir uçak tipini genelleme yoluyla öğrenmez. Kataloğa eklenmiş
bir referans kaydın kısa, gürültülü veya zaman kaydırılmış bir parçasını frekans
tepe çiftleri ve zaman farklarından oluşan hash'lerle bulur. Yeni fiziksel bir
uçağın tipini tahmin etmek öğrenilmiş modelin; kesin etiketi kabul etmek ise
insan doğrulamasının görevidir.

## Uygulamalar

Projede sunumda kullanılacak kısa giriş dosyaları bulunur:

```powershell
python app_shazam.py
python app_agent.py
python app_database.py
```

| Giriş | Görev |
|---|---|
| `app_shazam.py` | Ana sınıflandırma, üst sınıf ve Shazam/alt-tür akışı |
| `app_agent.py` | Çoklu ses modeli, altın referans ve insan-onay laboratuvarı |
| `app_database.py` | SQLite parmak izi veritabanını kurulum gerektirmeden salt okunur gösterir |

`app_shazam.py` arayüzünde önce **Üst Sınıfı Bul**, sonra **Shazam / Alt Türü
Bul** adımı kullanılır. Sonuç alanı alt türün Shazam eşleşmesinden mi yoksa
öğrenilmiş modelden mi geldiğini ayrıca gösterir.

`app_agent.py` akışı şöyledir:

1. Bağımsız test manifestinde bulunan bir ses seçilir.
2. BEATs, AST, PANNs/CNN14, CLAP, Fusion ve varsa fine-tune AST kanalları ayrı
   kanıt üretir.
3. Agent aday etiketi ve uzlaşma düzeyini gösterir.
4. Aday ses ile doğrulanmış altın referans ayrı ayrı dinlenir ve spektrogramları
   karşılaştırılır.
5. İnsan `Onayla`, `Reddet` veya `Emin Değilim` kararı verir.
6. Yalnız onaylı kayıtlar izole Shazam indeksine aktarılabilir.

Bu laboratuvarda Shazam tahmin üretmez; onaylanan referansları daha sonra
indekslemek için kullanılır. Böylece model önerisi ile kayıt eşleştirme kanıtı
birbirine karıştırılmaz.

## Hızlı devir teslim ve sunum testi

Projeyi ilk kez açan kişi, yerel `models/` ve veri klasörleri mevcutsa aşağıdaki
sırayla ilerleyebilir:

1. `python app_shazam.py` komutuyla ana uygulamayı açın.
2. `Test_Folder/SUNUM_TESTLERI/SHAZAM_TEST/` içinden dosya seçin; önce **Üst
   Sınıfı Bul**, ardından **Shazam / Alt Türü Bul** düğmesini kullanın.
3. `python app_agent.py` komutuyla insan-onay laboratuvarını açın.
4. `Test_Folder/SUNUM_TESTLERI/AGENT_TEST/` içinden dosya seçerek model
   kanallarını, altın referansı ve insan karar akışını gösterin.
5. SQLite içeriğini ek kurulum yapmadan incelemek için `python app_database.py`
   çalıştırın veya `SQLite_Goruntule.bat` dosyasını açın.

`Test_Folder/SUNUM_TESTLERI/manifest.json` sunum seslerinin beklenen kategori ve
alt tür bilgilerini taşır. Bu test sesleri eğitim referansı değildir ve aktif
Shazam indeksine eklenmemelidir. Tam yerel doğrulama için `python -m unittest
discover -v` komutu kullanılabilir.

## Kurulum

Desteklenen temel ortam: Windows 10/11, Python 3.10 veya 3.11. NVIDIA GPU
önerilir; GUI ve hafif modeller CPU üzerinde de açılabilir.

```powershell
git clone <REPO_URL>
cd Airport_Noise_Detection-main

python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

CUDA kullanılacaksa PyTorch paketlerini donanıma uygun komutla kurmak için
[PyTorch kurulum sayfasındaki](https://pytorch.org/get-started/locally/) güncel
komutu kullanın.

## Model ve veri artefaktları

Normal GitHub klonu kaynak kodu getirir. Büyük ikili dosyalar Git geçmişini
şişirmemesi ve GitHub'ın 100 MiB normal-dosya sınırına takılmaması için aynı
deponun **Releases** bölümünde, her biri 2 GiB'den küçük ZIP paketleri halinde
tutulur. Aşağıdaki yerel klasörler bu nedenle `.gitignore` kapsamındadır:

```text
models/       # ağırlıklar, encoder önbellekleri ve SQLite indeksleri
Self_Data/    # eğitim/referans sesleri
Test_Data/    # bağımsız test kayıtları
Test_Folder/  # sunum ve izole laboratuvar paketleri
downloads/    # yeniden indirilebilir ham veri setleri
cache/        # manifest ve embedding önbellekleri
outputs/      # yeniden üretilebilir rapor ve grafikler
```

### GitHub Release paketlerini oluşturma

Proje kökünde aşağıdaki komutu çalıştırın:

```powershell
python tools/prepare_release_assets.py
```

Komut `models/`, `Self_Data/`, `Test_Data/`, `Test_Folder/` ve `cache/`
içeriğini `release_assets/` altında yaklaşık 1,75 GiB'lik ZIP parçalarına
ayırır. Aynı klasörde oluşan `release_assets_manifest.json`, her paketin
SHA-256 özetini ve dosya sayısını taşır. ZIP'leri ve manifesti GitHub'da
**Releases > Draft a new release** ekranından aynı sürüme yükleyin.

`downloads/` yeniden indirilebilir ham kaynak aynalarını içerdiğinden varsayılan
pakete girmez. Kaynakların yeniden dağıtım lisansları tek tek doğrulandıysa
onları da dahil etmek için:

```powershell
python tools/prepare_release_assets.py --include-downloads
```

### Yeni klonda büyük dosyaları geri yükleme

Release sayfasındaki bütün ZIP parçalarını ve manifesti klon içindeki
`release_assets/` klasörüne indirin. Ardından:

```powershell
python tools/restore_release_assets.py
```

Araç önce bütün ZIP'lerin SHA-256 özetini doğrular, sonra dosyaları özgün
`models/`, `Self_Data/`, `Test_Data/`, `Test_Folder/` ve `cache/` yollarına
yerleştirir. Mevcut dosyaların üstüne yazmak gerekiyorsa açıkça
`--overwrite` kullanın. `venv/` hiçbir pakete dahil edilmez; bağımlılıklar
`requirements.txt` üzerinden yeniden kurulur.

Ana arayüz model dosyası bulunmadığında ilgili kanalı devre dışı bırakır ve
mevcut kanallarla devam eder. Parmak izi veritabanı için öncelik sırası:

1. `models/aircraft_fingerprints_3000.sqlite3`
2. `models/aircraft_fingerprints.sqlite3`

BEATs'in eski eğitim/inference yolu `C:\models` altında
`BEATs_iter3_plus_AS2M.pt` ve `beats_mlp.pt` bekler. Bu iki dosya yoksa BEATs
kanalı kapanır; EfficientNet/CNN/SVM ve yerel alt-tür kanalları çalışmaya devam
eder. Yeni kurulumlarda model artefaktlarını sürümlü bir model deposunda veya
Git LFS üzerinde tutmanız önerilir.

## Parmak izi indeksini üretme

Referansları `Self_Data/AIRCRAFT_TYPES/<UCAK_TIPI>/` düzeninde hazırlayın ve
manifest/indeks araçlarını çalıştırın:

```powershell
python build_aircraft_type_manifest.py
python build_aircraft_fingerprints.py --manifest cache/aircraft_type_manifest.csv --split train --rebuild
```

Tek kayıt eşleştirme testi:

```powershell
python match_aircraft.py test.wav
```

Veri sızıntısını önlemek için aynı fiziksel uçağa veya aynı kaynak kayda ait
parçalar train, validation ve test arasında bölünmemelidir. Test kayıtları aktif
Shazam indeksine eklenmemelidir.

## Eğitim araçları

Ana kategori modelleri:

```powershell
python dataset_builder.py
python train_efficientnet.py
python train_cnn.py
python train_aircraft_guard.py
```

Uçak ve diğer alt türler:

```powershell
python build_balanced_aircraft_clips.py
python build_aircraft_type_manifest.py
python train_aircraft_type_beats.py
python train_aircraft_type_beats.py --manifest cache/category_subtypes_350.csv --category TRAFFIC
python train_aircraft_type_beats.py --manifest cache/category_subtypes_350.csv --category OTHER
```

Kaggle/Colab deneylerinin yönergeleri ve sızıntısız fiziksel-uçak ayrımı için
`COLAB_UCAK_ALT_TUR_EGITIMI.md` dosyasını kullanın.

## Doğrulanmış deney özeti

Aşağıdaki değerler altı üst kategorili bağımsız testte (595 kayıt) alınmıştır.
Model seçimi yalnız validation Macro-F1 ile yapılmış; bağımsız test seçim için
kullanılmamıştır.

| Yöntem | Test doğruluğu | Test Macro-F1 |
|---|---:|---:|
| Yalnız masking | %71,60 | %68,34 |
| Yalnız contrastive | %72,10 | %68,82 |
| Masking + contrastive | %72,27 | %69,35 |
| Hibrit + dengesizlik sampler'ı | %73,28 | %70,66 |
| Hibrit + sampler + OTHER uzmanı (focal) | **%73,78** | **%71,00** |

Son uzman deneyindeki yaklaşık sınıf F1 değerleri: AIRCRAFT `%90`, AMBIENT
`%74`, OTHER `%59`, SPEECH `%83`, TRAFFIC `%78`, WIND `%42`. WIND test desteği
yalnız 10 kayıt olduğu için bu sınıfın değeri daha geniş bağımsız veriyle tekrar
ölçülmelidir. Bu tablo araştırma sonucudur; ilgili Kaggle model klasörü
`models/` altına kurulmadan masaüstü GUI otomatik olarak bu modeli kullanmaz.

## Testler

Kaynak kodun hızlı doğrulaması:

```powershell
python -m compileall -q app_shazam.py app_agent.py app_database.py gui_main.py noise_detector.py
python -m unittest -v test_window_voting test_aircraft_fingerprint test_dataset_catalog
```

Agent ve referans zinciri için:

```powershell
python -m unittest -v test_aircraft_reference_prediction_v1 test_aircraft_lab_manifest_integrity test_aircraft_reference_review_v1
```

Bazı testler yerel model/veri artefaktı gerektirir; eksik artefakt nedeniyle
atlanan test ile kod hatası aynı şey değildir.

## Proje yapısı

```text
.
|-- app_shazam.py                  # ana uygulama girişi
|-- app_agent.py                   # agent/insan-onay laboratuvarı
|-- app_database.py                # SQLite kanıt görüntüleyicisi
|-- gui_main.py                    # ana PyQt6 arayüzü
|-- noise_detector.py              # üst kategori ve model orkestrasyonu
|-- noise_detector_category_fp_v2.py
|-- aircraft_fingerprint.py        # Shazam tarzı parmak izi çekirdeği
|-- aircraft_reference_*.py        # güvenli referans kabul zinciri
|-- aircraft_audio_ensemble_v3.py  # gelişmiş agent kanalları
|-- edge_device/                   # uç cihaz/merkez prototipi
|-- models/README.md               # yerel model envanteri
|-- outputs/README.md              # çalışma çıktısı politikası
|-- requirements.txt
`-- README.md
```

## GitHub ve veri politikası

- `git add .` komutundan önce `git status --short` ile kapsamı kontrol edin.
- Ham sesleri, kişisel kayıtları, model ağırlıklarını ve SQLite indekslerini
  normal Git geçmişine eklemeyin; doğrulanmış büyük dosyaları Release ZIP'i
  olarak yayımlayın.
- Veri setinin kaynak URL'sini, lisansını, özgün etiketini ve fiziksel uçak
  kimliğini manifestte koruyun.
- İnsan onayı olmayan bir model önerisini kesin etiket olarak yayımlamayın.
- Shazam indeksine yalnız lisansı ve kaynağı doğrulanmış kabul kayıtlarını alın.
- `UNKNOWN` sonucunu zorla kaldırmak yerine yeni bağımsız veri toplayın.

## Bilinen sınırlamalar

- Sınıf dağılımı dengesizdir; özellikle WIND ve nadir uçak tiplerinde daha fazla
  bağımsız fiziksel kaynak gerekir.
- Parmak izi yalnız katalog kapsamındaki kayıtları tanır; görülmemiş sesleri
  genellemez.
- Alt tür modellerinin kapsamı yerel ağırlık dosyasındaki sınıflarla sınırlıdır.
- Büyük model ve veri Release paketleri indirilmeden yalnız kaynak kod ve hafif
  testler kullanılabilir.
