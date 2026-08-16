"""
=============================================================
 MAKINE ÖĞRENMESİ TABANLI SINIFLANDIRICI – Uzantı Modülü
=============================================================

Bu modül, kural tabanlı sınıflandırıcıya ek olarak
numpy/scipy ile sıfırdan implement edilmiş:

  1. GaussianNB    – Naive Bayes (hızlı, az veri yeterli)
  2. KNNClassifier – k-En Yakın Komşu
  3. MLNoiseClassifier – Eğitim + tahmin pipeline'ı

Gerçek veri olmadan önce, sistemi 'kural tabanlı etiketlerle
önyüklemek' (bootstrapping) mümkündür.

TensorFlow/PyTorch varsa CNN tabanlı sınıflandırma için
alt sınıf MirroredCNNClassifier kullanılabilir (taslak).
"""

import numpy as np
from noise_detector import FeatureExtractor, NoiseClassifier, AudioLoader, _synth_wav


# ═══════════════════════════════════════════════════════
#  Yardımcı: Özellik vektörü oluştur
# ═══════════════════════════════════════════════════════

def build_feature_vector(features, frame_idx: int) -> np.ndarray:
    """
    Tek bir zaman çerçevesi için sabit boyutlu özellik vektörü.

    İçerik: [sc, sb, sr, zcr, rms]  →  5 boyutlu
    """
    t_sc, sc  = features["spectral_centroid"]
    t_sb, sb  = features["spectral_bandwidth"]
    t_sr, sr_ = features["spectral_rolloff"]
    t_zcr, zcr= features["zcr"]
    t_rms, rms= features["rms"]

    n = min(len(sc), len(sb), len(sr_), len(zcr), len(rms))
    if frame_idx >= n:
        frame_idx = n - 1

    vec = np.array([
        sc[frame_idx],
        sb[frame_idx],
        sr_[frame_idx],
        zcr[frame_idx],
        rms[frame_idx],
    ], dtype=np.float32)
    return vec


def build_dataset_from_features(features, rule_labels):
    """
    Kural tabanlı etiketlerden (bootstrapping) eğitim seti oluştur.

    Returns
    -------
    X : (n_frames, 5) float32
    y : (n_frames,)  int  – label index
    """
    label_to_idx = {l: i for i, l in enumerate(NoiseClassifier.LABELS)}

    n = len(rule_labels)
    X = np.zeros((n, 5), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)

    for i in range(n):
        X[i] = build_feature_vector(features, i)
        y[i] = label_to_idx.get(rule_labels[i], 4)

    return X, y


# ═══════════════════════════════════════════════════════
#  Gaussian Naive Bayes (numpy ile)
# ═══════════════════════════════════════════════════════

class GaussianNaiveBayes:
    """Sıfırdan implement edilmiş Gaussian Naive Bayes."""

    def fit(self, X, y):
        self.classes_    = np.unique(y)
        self.n_classes_  = len(self.classes_)
        n_features       = X.shape[1]

        self.priors_ = np.zeros(self.n_classes_)
        self.means_  = np.zeros((self.n_classes_, n_features))
        self.vars_   = np.zeros((self.n_classes_, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[i] = len(X_c) / len(X)
            self.means_[i]  = X_c.mean(axis=0)
            self.vars_[i]   = X_c.var(axis=0) + 1e-9  # Laplace smoothing

        return self

    def _log_likelihood(self, X):
        """Log-olasılık matrisi: (n_samples, n_classes)"""
        log_probs = np.zeros((len(X), self.n_classes_))
        for i in range(self.n_classes_):
            # Gaussian log p(x|c)
            log_p = -0.5 * np.sum(
                np.log(2 * np.pi * self.vars_[i]) +
                ((X - self.means_[i]) ** 2) / self.vars_[i],
                axis=1
            )
            log_probs[:, i] = log_p + np.log(self.priors_[i])
        return log_probs

    def predict(self, X):
        log_probs = self._log_likelihood(X)
        return self.classes_[np.argmax(log_probs, axis=1)]

    def predict_proba(self, X):
        log_probs = self._log_likelihood(X)
        # Softmax normalize
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════
#  k-En Yakın Komşu
# ═══════════════════════════════════════════════════════

class KNNClassifier:
    """Sıfırdan implement edilmiş k-NN sınıflandırıcı."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        # Normalize
        self.X_mean_ = X.mean(axis=0)
        self.X_std_  = X.std(axis=0) + 1e-9
        self.X_train_= (X - self.X_mean_) / self.X_std_
        self.y_train_= y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X_norm = (X - self.X_mean_) / self.X_std_
        preds  = np.zeros(len(X), dtype=np.int32)

        for i, x in enumerate(X_norm):
            # Öklidyen mesafe
            dists    = np.sqrt(np.sum((self.X_train_ - x) ** 2, axis=1))
            k_idx    = np.argsort(dists)[:self.k]
            k_labels = self.y_train_[k_idx]
            # Çoğunluk oyu
            preds[i] = np.bincount(k_labels,
                                   minlength=len(self.classes_)).argmax()
        return preds


# ═══════════════════════════════════════════════════════
#  MLNoiseClassifier – Pipeline
# ═══════════════════════════════════════════════════════

class MLNoiseClassifier:
    """
    Makine öğrenmesi tabanlı gürültü sınıflandırıcı.

    1. Kural tabanlı etiketlerle başlangıç (bootstrapping)
    2. GaussianNB veya KNN ile eğitim
    3. Tahmin + güven skoru
    """

    LABELS = NoiseClassifier.LABELS

    def __init__(self, model_type="gnb", k_neighbors=7):
        self.model_type = model_type
        if model_type == "gnb":
            self.model = GaussianNaiveBayes()
        elif model_type == "knn":
            self.model = KNNClassifier(k=k_neighbors)
        else:
            raise ValueError(f"Bilinmeyen model: {model_type}")

        self._trained = False

    def bootstrap_train(self, samples, sr=22050):
        """
        Sesin kural tabanlı etiketlerini kullanarak modeli eğit.
        Gerçek etiketli veri yokken kullanışlıdır.
        """
        extractor   = FeatureExtractor(sr=sr)
        rule_clf    = NoiseClassifier()

        features     = extractor.extract_all(samples)
        rule_labels, _, _ = rule_clf.classify(features)

        X, y = build_dataset_from_features(features, rule_labels)

        # Eğitim / test ayırımı (%80 / %20)
        n_train = int(0.8 * len(X))
        idx     = np.random.permutation(len(X))
        X_tr, y_tr = X[idx[:n_train]], y[idx[:n_train]]
        X_te, y_te = X[idx[n_train:]], y[idx[n_train:]]

        self.model.fit(X_tr, y_tr)
        self._trained = True

        # Doğruluk hesapla
        y_pred   = self.model.predict(X_te)
        accuracy = np.mean(y_pred == y_te) * 100
        print(f"[MLClassifier] Model: {self.model_type.upper()}  |  "
              f"Test doğruluğu (kural etiketleri): {accuracy:.1f}%  |  "
              f"Eğitim örnekleri: {n_train}")
        return accuracy

    def predict_segment(self, features, frame_idx: int):
        """Tek bir çerçeveyi tahmin et."""
        if not self._trained:
            raise RuntimeError("Önce bootstrap_train() çalıştırın.")
        vec  = build_feature_vector(features, frame_idx).reshape(1, -1)
        pred = self.model.predict(vec)[0]
        return self.LABELS[pred]

    def predict_all(self, features):
        """Tüm çerçeveleri tahmin et."""
        if not self._trained:
            raise RuntimeError("Önce bootstrap_train() çalıştırın.")

        n = len(features["spectral_centroid"][0])
        X = np.array([build_feature_vector(features, i) for i in range(n)])
        preds   = self.model.predict(X)
        labels  = [self.LABELS[p] for p in preds]
        return labels


# ═══════════════════════════════════════════════════════
#  CNN Taslak (TensorFlow / PyTorch)
# ═══════════════════════════════════════════════════════

CNN_ARCHITECTURE_NOTES = """
CNN Tabanlı Uçak Sesi Sınıflandırıcı – Taslak

Giriş: Mel spektrogram  →  (128 mel bin, N zaman adımı, 1 kanal)

Katmanlar:
  Conv2D(32, 3x3)  → BatchNorm → ReLU → MaxPool(2x2)
  Conv2D(64, 3x3)  → BatchNorm → ReLU → MaxPool(2x2)
  Conv2D(128, 3x3) → BatchNorm → ReLU → GlobalAvgPool
  Dense(256)       → Dropout(0.4)
  Dense(5)         → Softmax

Eğitim:
  - Veri kümesi: UrbanSound8K + özel havalimanı kayıtları
  - Augmentation: zaman kayması, pitch shifting, gürültü ekleme
  - Optimizer: Adam(lr=1e-4)
  - Loss: Categorical Crossentropy

TensorFlow örneği:
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, (3,3), activation='relu',
                             input_shape=(128, 128, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.Dense(5, activation='softmax')
  ])
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
"""


# ═══════════════════════════════════════════════════════
#  Demo
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    print("\n" + "=" * 55)
    print("  ML SINIFLANDIRICI DEMO")
    print("=" * 55)

    # Sentetik veri üret
    demo_path = "ml_demo.wav"
    _synth_wav(demo_path, duration=8.0, sr=22050)

    loader   = AudioLoader(target_sr=22050)
    samples, sr = loader.load(demo_path)

    # ─ GNB sınıflandırıcı ────────────────────────────
    print("\n[1] Gaussian Naive Bayes")
    gnb = MLNoiseClassifier(model_type="gnb")
    gnb.bootstrap_train(samples, sr)

    extractor = FeatureExtractor(sr=sr)
    feats     = extractor.extract_all(samples)
    gnb_labels = gnb.predict_all(feats)

    from collections import Counter
    print("  GNB Dağılımı:", dict(Counter(gnb_labels)))

    # ─ KNN sınıflandırıcı ────────────────────────────
    print("\n[2] k-NN (k=7)")
    knn = MLNoiseClassifier(model_type="knn", k_neighbors=7)
    knn.bootstrap_train(samples, sr)
    knn_labels = knn.predict_all(feats)
    print("  KNN Dağılımı:", dict(Counter(knn_labels)))

    print("\n" + CNN_ARCHITECTURE_NOTES)

    # Temizlik
    if os.path.exists(demo_path):
        os.remove(demo_path)
