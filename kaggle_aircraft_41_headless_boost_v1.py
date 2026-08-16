"""Headless AST embeddings + CatBoost/XGBoost/LightGBM for 41 aircraft types."""
from __future__ import annotations

import json, random, shutil
from collections import Counter
from pathlib import Path

import joblib, librosa, numpy as np, pandas as pd, torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from xgboost import XGBClassifier

SEED, SR, SECONDS = 42, 16000, 10
BASE_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/kaggle/working/aircraft_41_headless_boost_v1")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type != "cuda": raise RuntimeError("Kaggle GPU acilmadi")


def locate_root():
    found = list(Path("/kaggle/input").rglob("AIRCRAFT_AST_41CLASS_V1/manifest.csv"))
    if not found: raise FileNotFoundError("AIRCRAFT_AST_41CLASS_V1 Add Input ile eklenmedi")
    return found[0].parent


root = locate_root(); OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(root / "manifest.csv")
df["path"] = df.path.map(lambda x: str(root / str(x)))
df["physical_airframe_id"] = df.physical_airframe_id.fillna("").astype(str)
labels = sorted(df.label.unique()); label2id = {x:i for i,x in enumerate(labels)}
id2label = {i:x for x,i in label2id.items()}; df["target"] = df.label.map(label2id)
assert len(labels) == 41

# Same physical aircraft can never occur in more than one split.
rng = random.Random(SEED); pieces=[]; coverage=[]
for label, group in df.groupby("label"):
    ids=sorted(group.physical_airframe_id.unique()); rng.shuffle(ids)
    if len(ids)>=5: nt=max(1,round(len(ids)*.15)); nv=max(1,round(len(ids)*.15))
    elif len(ids)>=3: nt=nv=1
    else: nt=nv=0
    test=set(ids[:nt]); val=set(ids[nt:nt+nv])
    g=group.copy(); g["split"]=g.physical_airframe_id.map(
        lambda x:"test" if x in test else "validation" if x in val else "train")
    pieces.append(g); coverage.append({"label":label,"recordings":len(g),"airframes":len(ids),
                                      "independent_test_ready":len(ids)>=3})
df=pd.concat(pieces,ignore_index=True); df.to_csv(OUT/"split_manifest.csv",index=False)

extractor=AutoFeatureExtractor.from_pretrained(BASE_MODEL)
ast=ASTForAudioClassification.from_pretrained(BASE_MODEL, attn_implementation="eager").to(DEVICE).eval()
for p in ast.parameters(): p.requires_grad=False


def crops(path):
    y,_=librosa.load(path,sr=SR,mono=True); n=SR*SECONDS
    if len(y)<=n: return [np.pad(y,(0,n-len(y))).astype(np.float32)]
    starts=sorted(set([0,(len(y)-n)//2,len(y)-n]))
    return [y[s:s+n].astype(np.float32) for s in starts]


@torch.no_grad()
def embedding(path):
    audio=crops(path)
    values=extractor(audio,sampling_rate=SR,return_tensors="pt")["input_values"].to(DEVICE)
    hidden=ast.audio_spectrogram_transformer(input_values=values).last_hidden_state
    pooled=(hidden[:,0]+hidden[:,1])/2.0
    pooled=torch.nn.functional.normalize(pooled,dim=1).mean(0)
    pooled=torch.nn.functional.normalize(pooled,dim=0)
    return pooled.cpu().numpy().astype(np.float32)


cache=OUT/"ast_headless_embeddings.npy"
if cache.exists(): X=np.load(cache)
else:
    vectors=[]
    for i,path in enumerate(df.path,1):
        vectors.append(embedding(path))
        if i%25==0 or i==len(df): print(f"Embedding {i}/{len(df)}")
    X=np.stack(vectors); np.save(cache,X)

tr=np.where(df.split.values=="train")[0]; va=np.where(df.split.values=="validation")[0]
te=np.where(df.split.values=="test")[0]
y=df.target.to_numpy(); ytr,yva,yte=y[tr],y[va],y[te]

# PCA is fitted on train only: 625 recordings versus 768 features otherwise overfits badly.
n_components=min(128,len(tr)-1,X.shape[1])
pca=PCA(n_components=n_components,whiten=True,random_state=SEED)
Xtr=pca.fit_transform(X[tr]); Xva=pca.transform(X[va]); Xte=pca.transform(X[te])
joblib.dump(pca,OUT/"pca.joblib")

weights=compute_sample_weight("balanced",ytr)
weights=np.clip(weights,.25,np.quantile(weights,.90))

models={
 "CatBoost":CatBoostClassifier(iterations=900,depth=6,learning_rate=.035,loss_function="MultiClass",
                                 random_seed=SEED,verbose=100,allow_writing_files=False,task_type="GPU"),
 "XGBoost":XGBClassifier(n_estimators=900,max_depth=6,learning_rate=.035,subsample=.85,
                           colsample_bytree=.8,min_child_weight=2,reg_lambda=2,reg_alpha=.05,
                           objective="multi:softprob",num_class=len(labels),eval_metric="mlogloss",
                           tree_method="hist",device="cuda",random_state=SEED),
 "LightGBM":LGBMClassifier(n_estimators=900,num_leaves=31,max_depth=-1,learning_rate=.03,
                             subsample=.85,colsample_bytree=.8,reg_lambda=2,reg_alpha=.05,
                             objective="multiclass",num_class=len(labels),verbosity=-1,random_state=SEED),
}


def aligned_probability(model,x):
    raw=model.predict_proba(x); out=np.zeros((len(x),len(labels)),dtype=np.float64)
    for source_index,class_id in enumerate(model.classes_): out[:,int(class_id)]=raw[:,source_index]
    return out


val_prob={}; test_prob={}; results=[]
for name,model in models.items():
    print("\nTRAIN",name); model.fit(Xtr,ytr,sample_weight=weights)
    joblib.dump(model,OUT/f"{name.lower()}.joblib")
    val_prob[name]=aligned_probability(model,Xva); test_prob[name]=aligned_probability(model,Xte)
    vp=val_prob[name].argmax(1); tp=test_prob[name].argmax(1)
    results.append({"model":name,"validation_macro_f1":f1_score(yva,vp,average="macro",zero_division=0),
                    "test_accuracy":accuracy_score(yte,tp),
                    "test_macro_f1":f1_score(yte,tp,average="macro",zero_division=0)})

# Validation Macro-F1 determines ensemble weights; test labels never tune the ensemble.
vf={r["model"]:max(r["validation_macro_f1"],1e-4) for r in results}; total=sum(vf.values())
ensemble_weights={k:v/total for k,v in vf.items()}
ens_val=sum(ensemble_weights[k]*val_prob[k] for k in models)
ens_test=sum(ensemble_weights[k]*test_prob[k] for k in models)
vp=ens_val.argmax(1); tp=ens_test.argmax(1)
results.append({"model":"WeightedSoftVote","validation_macro_f1":f1_score(yva,vp,average="macro",zero_division=0),
                "test_accuracy":accuracy_score(yte,tp),"test_macro_f1":f1_score(yte,tp,average="macro",zero_division=0)})

# Recording and physical-aircraft metrics for the final ensemble.
pred=df.iloc[te][["path","label","physical_airframe_id"]].copy()
pred["truth_id"]=yte; pred["prediction_id"]=tp; pred["prediction_label"]=[id2label[i] for i in tp]
pred["confidence"]=ens_test.max(1); pred.to_csv(OUT/"test_predictions.csv",index=False)
air_y=[]; air_p=[]
for _,g in pred.groupby("physical_airframe_id"):
    air_y.append(Counter(g.truth_id).most_common(1)[0][0]); air_p.append(Counter(g.prediction_id).most_common(1)[0][0])

report={"classes_in_output":41,"method":"Frozen headless AST embeddings + PCA + boosted trees",
        "pca_dimensions":n_components,"ensemble_weights":ensemble_weights,"model_results":results,
        "ensemble_test_airframe_accuracy":accuracy_score(air_y,air_p),
        "ensemble_test_airframe_macro_f1":f1_score(air_y,air_p,average="macro",zero_division=0),
        "scope_warning":"23 rare types still lack >=3 physical airframes; no model can independently validate them."}
(OUT/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
(OUT/"classification_report.txt").write_text(classification_report(yte,tp,labels=list(range(41)),
 target_names=labels,zero_division=0),encoding="utf-8")
(OUT/"labels.json").write_text(json.dumps({"label2id":label2id,"id2label":id2label},indent=2),encoding="utf-8")
(OUT/"coverage_report.json").write_text(json.dumps(coverage,indent=2),encoding="utf-8")
shutil.make_archive("/kaggle/working/aircraft_41_headless_boost_v1","zip",OUT)
print(json.dumps(report,indent=2)); print("ZIP: /kaggle/working/aircraft_41_headless_boost_v1.zip")
