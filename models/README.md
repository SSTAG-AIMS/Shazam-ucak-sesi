# Yerel model dosyaları

Bu klasördeki ağırlıklar, embedding dosyaları ve SQLite parmak izi indeksleri
normal Git geçmişine eklenmez. `.gitignore`, bu dosyaları yanlışlıkla Git
nesnesi olarak yüklenmeye karşı korur. Paylaşım lisansı doğrulanan artefaktlar
ana README'de açıklanan GitHub Release ZIP paketlerinde yayımlanır.

Ana arayüz kullanılabilir dosyaları başlangıçta otomatik keşfeder. Tam yerel
kurulumda başlıca artefaktlar şunlardır:

- `best_efficientnet.pt`, `efficientnet_label_encoder.pkl`
- `best_cnn.pt`, `cnn_label_encoder.pkl`
- `best_model.pkl`, `label_encoder.pkl`
- `aircraft_guard.pkl`
- `aircraft_type_beats.pt`
- `traffic_subtype_beats.pt`, `other_subtype_beats.pt`
- `aircraft_fingerprints_3000.sqlite3` veya `aircraft_fingerprints.sqlite3`
- `aircraft_ast_finetuned_v4/` (agent denetim kanalı)
- `aircraft_audio_ensemble_v3.joblib` ve isteğe bağlı PANNs/AST önbelleği

Model dosyalarını Release paketlerinden `python tools/restore_release_assets.py`
ile geri yükleyin veya ana README'deki eğitim/indeks komutlarıyla yeniden
üretin. Büyük artefaktları normal Git geçmişine eklemeyin.
