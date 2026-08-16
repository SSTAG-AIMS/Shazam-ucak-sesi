# Colab: uçak alt türü AST fine-tuning V1

Bu çalışma `AIRBUS_A320`, `BOEING_737_800` gibi uçak tiplerini tahmin eder.
Shazam değildir. Aynı kaydı aramak yerine daha önce görmediği fiziksel uçaklara
genelleme yapmaya çalışır.

## Bilimsel koşullar

- Aynı fiziksel uçak (`hex_id`) yalnız bir split içinde bulunur.
- Aynı ana kayıttan kesilen pencereler train ve test arasında paylaşılmaz.
- Bir sınıf en az 12, tercihen 30–50 farklı ana kayda ulaşmadan üretim sınıfı sayılmaz.
- Test kümesi eğitim ve eşik ayarında hiçbir zaman kullanılmaz.
- Model garantili sonuç vermez; accuracy, macro-F1 ve confusion matrix ile raporlanır.

## 1. Colab GPU ve paketler

Colab'da `Runtime > Change runtime type > T4 GPU` seçin.

```python
!pip -q install "transformers==4.46.3" "accelerate==1.1.1" \
    "librosa==0.10.2.post1" "soundfile==0.12.1" \
    "scikit-learn==1.5.2" "pandas==2.2.3"
```

## 2. Veriyi yükleme

Bilgisayarda `Self_Data/AIRCRAFT_100_REFERENCE_V1` klasörünü ZIP yapıp Colab'a
yükleyin. Klasör yapısı şöyle olmalıdır:

```text
AIRCRAFT_100_REFERENCE_V1/
  AIRBUS_A320/*.wav
  BOEING_737_800/*.wav
  ...
```

```python
from google.colab import files
uploaded = files.upload()  # ZIP dosyasını seç
zip_name = next(iter(uploaded))
!rm -rf /content/aircraft_data
!mkdir -p /content/aircraft_data
!unzip -q "$zip_name" -d /content/aircraft_data
```

## 3. Kaynak kayıt ve fiziksel uçak bazlı split

Dosya adının ilk bölümü ADS-B `hex_id` olarak kullanılır. Aşağıdaki kod aynı
uçağın train/test sızıntısı yapmasını engeller.

```python
from pathlib import Path
import random, pandas as pd

ROOT = Path('/content/aircraft_data/AIRCRAFT_100_REFERENCE_V1')
if not ROOT.exists():
    ROOT = next(Path('/content/aircraft_data').rglob('AIRCRAFT_100_REFERENCE_V1'))

rows = []
for label_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    for path in sorted(label_dir.glob('*.wav')):
        rows.append({
            'path': str(path),
            'label': label_dir.name,
            'airframe': path.stem.split('_')[0].upper(),
        })
df = pd.DataFrame(rows)

# Sunum deneyi için 5 yapılabilir; savunulabilir model için 12+, ideal 30–50.
MIN_INDEPENDENT_RECORDINGS = 12
counts = df.groupby('label')['airframe'].nunique()
valid = counts[counts >= MIN_INDEPENDENT_RECORDINGS].index
df = df[df.label.isin(valid)].copy()
assert len(valid) >= 2, (
    'Yeterli sınıf yok. Eşiği yapay olarak düşürmek yerine her uçak tipi için '
    'daha fazla bağımsız kaynak kayıt toplayın.'
)

rng = random.Random(42)
parts = []
for label, group in df.groupby('label'):
    airframes = sorted(group.airframe.unique())
    rng.shuffle(airframes)
    n = len(airframes)
    n_test = max(2, round(n * .20))
    n_val = max(2, round(n * .20))
    test_ids = set(airframes[:n_test])
    val_ids = set(airframes[n_test:n_test+n_val])
    current = group.copy()
    current['split'] = current.airframe.map(
        lambda x: 'test' if x in test_ids else ('validation' if x in val_ids else 'train')
    )
    parts.append(current)
df = pd.concat(parts, ignore_index=True)

print(df.groupby(['label','split']).size().unstack(fill_value=0))
for airframe, group in df.groupby('airframe'):
    assert group.split.nunique() == 1
df.to_csv('/content/aircraft_split_manifest.csv', index=False)
```

## 4. AST veri kümesi

Her ana kayıt eğitim sırasında farklı bir 10 saniyelik bölüm verir. Split ana
kayıt düzeyinde yapıldığı için bu pencereleme veri sızıntısı oluşturmaz.

```python
import numpy as np, librosa, torch
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor

MODEL_ID = 'MIT/ast-finetuned-audioset-10-10-0.4593'
processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
labels = sorted(df.label.unique())
label2id = {v:i for i,v in enumerate(labels)}
id2label = {i:v for v,i in label2id.items()}

class AircraftDataset(Dataset):
    def __init__(self, frame, train=False):
        self.frame = frame.reset_index(drop=True)
        self.train = train
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        y, _ = librosa.load(row.path, sr=16000, mono=True)
        target = 16000 * 10
        if len(y) > target:
            if self.train:
                start = np.random.randint(0, len(y)-target+1)
            else:
                start = (len(y)-target)//2
            y = y[start:start+target]
        else:
            y = np.pad(y, (0, target-len(y)))
        inputs = processor(y, sampling_rate=16000, return_tensors='pt')
        return {
            'input_values': inputs['input_values'].squeeze(0),
            'labels': torch.tensor(label2id[row.label]),
        }

train_ds = AircraftDataset(df[df.split=='train'], train=True)
val_ds = AircraftDataset(df[df.split=='validation'])
test_ds = AircraftDataset(df[df.split=='test'])
```

## 5. Modeli gerçekten fine-tune etme

Burada yalnız SVM başlığı değil, AST sınıflandırma katmanı ve son encoder
katmanları uçak verisine uyarlanır.

```python
from transformers import AutoModelForAudioClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

# Küçük veri için alt katmanları dondur, son iki encoder bloğunu eğit.
for p in model.audio_spectrogram_transformer.parameters():
    p.requires_grad = False
for block in model.audio_spectrogram_transformer.encoder.layer[-2:]:
    for p in block.parameters(): p.requires_grad = True
for p in model.classifier.parameters(): p.requires_grad = True

def metrics(result):
    truth = result.label_ids
    pred = result.predictions.argmax(axis=1)
    return {
        'accuracy': accuracy_score(truth, pred),
        'macro_f1': f1_score(truth, pred, average='macro', zero_division=0),
    }

args = TrainingArguments(
    output_dir='/content/ast_aircraft_runs',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=25,
    warmup_ratio=.10,
    weight_decay=.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to='none',
    seed=42,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=metrics,
)
trainer.train()
```

## 6. Testi yalnız bir kez açma ve kanıt üretme

```python
import json
from sklearn.metrics import classification_report, confusion_matrix

prediction = trainer.predict(test_ds)
truth = prediction.label_ids
pred = prediction.predictions.argmax(axis=1)
report = {
    'metrics': metrics(prediction),
    'classification_report': classification_report(
        truth, pred, labels=list(range(len(labels))), target_names=labels,
        output_dict=True, zero_division=0,
    ),
    'confusion_matrix': confusion_matrix(
        truth, pred, labels=list(range(len(labels)))
    ).tolist(),
    'test_files': len(test_ds),
    'split_rule': 'physical_airframe_disjoint',
}
print(json.dumps(report['metrics'], indent=2))
```

## 7. Projeye aktarılacak paket

```python
OUT = '/content/aircraft_ast_finetuned_v1'
trainer.save_model(OUT)
processor.save_pretrained(OUT)
with open(f'{OUT}/independent_test_report.json','w') as f:
    json.dump(report, f, indent=2)
df.to_csv(f'{OUT}/split_manifest.csv', index=False)
!cd /content && zip -qr aircraft_ast_finetuned_v1.zip aircraft_ast_finetuned_v1
files.download('/content/aircraft_ast_finetuned_v1.zip')
```

ZIP açıldığında proje yolu şu olmalıdır:

```text
C:\Airport_Noise_Detection-main\models\aircraft_ast_finetuned_v1\
  config.json
  model.safetensors
  preprocessor_config.json
  split_manifest.csv
  independent_test_report.json
```

Bu klasörü projeye koyduktan sonra inference adaptörü ve agent karar politikasını
bu yeni pakete bağlamak gerekir. Eski `aircraft_audio_ensemble_v3.joblib`
dosyasının üzerine yazmayın; geri dönüş için eski sistem yerinde kalmalıdır.
