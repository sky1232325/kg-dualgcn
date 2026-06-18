"""
Complete GPU experiment suite for the paper.
Runs ALL experiments on the expanded dataset (3678 samples, 737 test).
Produces a single JSON file with all results needed for the paper tables.

Experiments:
  1. Main comparison: BERT Only, R-GCN, Concat, DualChannel
  2. Ablation: Add fusion, Gate fusion, no class weights, no KG, no text
  3. Hyperparameter sensitivity: LR × class weight grid
  4. 5-fold cross-validation with paired t-test
  5. Robustness: 10%, 20%, 30% entity drop
  6. K-BERT baseline

Usage: python run_all_gpu_experiments.py
"""
import os, sys, json, random, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import RGCNConv
from scipy import stats

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)
t_start = time.time()

MODEL_NAME = "C:/Users/liuchun/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/clean"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
DATA_DIR = "D:/code/Hermes/ad-dualchannel-gcn/data/processed"
CACHE_DIR = "D:/code/Hermes/ad-dualchannel-gcn/data/cache"

def load_json(p):
    with open(p, 'r', encoding='utf-8') as f: return json.load(f)

train_data = load_json(f"{DATA_DIR}/train.json")
val_data = load_json(f"{DATA_DIR}/val.json")
test_data = load_json(f"{DATA_DIR}/test.json")
print(f"Data: train={len(train_data)} val={len(val_data)} test={len(test_data)}", flush=True)

# ============================================================
# Dataset & Collate
# ============================================================
class CachedDS(Dataset):
    def __init__(self, samples, key='labels'):
        self.samples = samples; self.key = key
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.key == 'labels' and 'label' in s and 'labels' not in s:
            s = dict(s); s['labels'] = s.pop('label')
        return s

def collate(batch):
    B = len(batch); mx = max(i["node_features"].size(0) for i in batch)
    ids = torch.stack([i["input_ids"] for i in batch])
    am = torch.stack([i["attention_mask"] for i in batch])
    labs = torch.stack([i["labels"] for i in batch])
    d = batch[0]["node_features"].size(1)
    pnf = torch.zeros(B*mx, d); nb = torch.zeros(B*mx, dtype=torch.long)
    for i, it in enumerate(batch):
        n = it["node_features"].size(0)
        pnf[i*mx:i*mx+n] = it["node_features"]; nb[i*mx:i*mx+n] = i
    offs = torch.cumsum(torch.tensor([0]+[i["node_features"].size(0) for i in batch[:-1]]), 0)
    asr, adt, at = [], [], []
    for i, it in enumerate(batch):
        if it["edge_index"].size(1) > 0:
            asr.append(it["edge_index"][0]+offs[i]); adt.append(it["edge_index"][1]+offs[i]); at.append(it["edge_type"])
    if asr: ei = torch.stack([torch.cat(asr), torch.cat(adt)]); et = torch.cat(at)
    else: ei = torch.zeros(2, 0, dtype=torch.long); et = torch.zeros(0, dtype=torch.long)
    return {"input_ids":ids,"attention_mask":am,"node_features":pnf,"edge_index":ei,"edge_type":et,"node_batch":nb,"labels":labs}

def kb_collate(batch):
    ids = torch.stack([i["input_ids"] for i in batch])
    am = torch.stack([i["attention_mask"] for i in batch])
    labs = torch.stack([i["labels"] for i in batch])
    return {"input_ids":ids,"attention_mask":am,"labels":labs}

# ============================================================
# Models
# ============================================================
class RG(nn.Module):
    def __init__(s,ind,hid,outd,nr,nl=2,dp=0.2):
        super().__init__()
        s.ls=nn.ModuleList([RGCNConv(ind,hid,nr)]+[RGCNConv(hid,hid,nr) for _ in range(nl-2)]+[RGCNConv(hid,outd,nr)])
        s.dp=nn.Dropout(dp)
    def forward(s,x,ei,et):
        for i,l in enumerate(s.ls):
            x=l(x,ei,et)
            if i<len(s.ls)-1: x=F.relu(x); x=s.dp(x)
        return x

def pg(ne,nb,B):
    ge=[]
    for i in range(B):
        m=nb==i; ge.append(ne[m].mean(0) if m.sum()>0 else torch.zeros(ne.size(1),device=ne.device))
    return torch.stack(ge)

class BiCA(nn.Module):
    def __init__(s,dt,dg,fd,nh=4,dp=0.1):
        super().__init__()
        s.tp=nn.Linear(dt,fd); s.gp=nn.Linear(dg,fd)
        s.t2g=nn.MultiheadAttention(fd,nh,dropout=dp,batch_first=True)
        s.g2t=nn.MultiheadAttention(fd,nh,dropout=dp,batch_first=True)
        s.n1=nn.LayerNorm(fd); s.n2=nn.LayerNorm(fd); s.out=nn.Linear(fd*2,fd)
    def forward(s,ts,gn):
        T=s.tp(ts); G=s.gp(gn)
        t2g,_=s.t2g(T,G,G); t2g=s.n1(T+t2g)
        g2t,_=s.g2t(G,T,T); g2t=s.n2(G+g2t)
        return s.out(torch.cat([t2g.mean(1),g2t.mean(1)],-1))

class ConcatFusion(nn.Module):
    def __init__(s,dt,dg,fd,dp=0.1):
        super().__init__()
        s.tp=nn.Linear(dt,fd); s.gp=nn.Linear(dg,fd)
        s.out=nn.Linear(fd*2,fd)
    def forward(s,ts,gn):
        T=s.tp(ts).mean(1); G=s.gp(gn).mean(1)
        return s.out(torch.cat([T,G],-1))

class AddFusion(nn.Module):
    def __init__(s,dt,dg,fd,dp=0.1):
        super().__init__()
        s.tp=nn.Linear(dt,fd); s.gp=nn.Linear(dg,fd)
    def forward(s,ts,gn):
        T=s.tp(ts).mean(1); G=s.gp(gn).mean(1)
        return T+G

class GateFusion(nn.Module):
    def __init__(s,dt,dg,fd,dp=0.1):
        super().__init__()
        s.tp=nn.Linear(dt,fd); s.gp=nn.Linear(dg,fd)
        s.gate=nn.Linear(fd*2,fd)
    def forward(s,ts,gn):
        T=s.tp(ts).mean(1); G=s.gp(gn).mean(1)
        g=torch.sigmoid(s.gate(torch.cat([T,G],-1)))
        return g*T+(1-g)*G

class DualChannel(nn.Module):
    def __init__(s,bn,nc=2,gh=256,fd=256,dp=0.2,fusion='crossattn'):
        super().__init__()
        s.bert=BertModel.from_pretrained(bn); bd=s.bert.config.hidden_size
        s.rgcn=RG(768,gh,gh,1,nl=2,dp=dp)
        if fusion=='crossattn':
            s.fus=BiCA(bd,gh,fd,nh=4,dp=dp)
        elif fusion=='concat':
            s.fus=ConcatFusion(bd,gh,fd,dp)
        elif fusion=='add':
            s.fus=AddFusion(bd,gh,fd,dp)
        elif fusion=='gate':
            s.fus=GateFusion(bd,gh,fd,dp)
        s.cls=nn.Sequential(nn.Linear(fd,128),nn.ReLU(),nn.Dropout(dp),nn.Linear(128,nc))
    def forward(s,b):
        bo=s.bert(input_ids=b["input_ids"],attention_mask=b["attention_mask"])
        ne=s.rgcn(b["node_features"],b["edge_index"],b["edge_type"])
        ge=pg(ne,b["node_batch"],b["input_ids"].size(0)).unsqueeze(1)
        f=s.fus(bo.last_hidden_state,ge); return s.cls(f),f

class BERTOnly(nn.Module):
    def __init__(s,bn,nc=2,dp=0.2):
        super().__init__()
        s.bert=BertModel.from_pretrained(bn); bd=s.bert.config.hidden_size
        s.cls=nn.Sequential(nn.Linear(bd,128),nn.ReLU(),nn.Dropout(dp),nn.Linear(128,nc))
    def forward(s,b):
        bo=s.bert(input_ids=b["input_ids"],attention_mask=b["attention_mask"])
        return s.cls(bo.last_hidden_state[:,0,:]), bo.last_hidden_state[:,0,:]

class RGCNOnly(nn.Module):
    def __init__(s,nc=2,gh=256,fd=256,dp=0.2):
        super().__init__()
        s.rgcn=RG(768,gh,gh,1,nl=2,dp=dp)
        s.cls=nn.Sequential(nn.Linear(gh,128),nn.ReLU(),nn.Dropout(dp),nn.Linear(128,nc))
    def forward(s,b):
        ne=s.rgcn(b["node_features"],b["edge_index"],b["edge_type"])
        ge=pg(ne,b["node_batch"],b["input_ids"].size(0))
        return s.cls(ge), ge

class KBERTModel(nn.Module):
    def __init__(s, bn, nc=2, dp=0.2):
        super().__init__()
        s.bert = BertModel.from_pretrained(bn); bd = s.bert.config.hidden_size
        s.cls = nn.Sequential(nn.Linear(bd, 128), nn.ReLU(), nn.Dropout(dp), nn.Linear(128, nc))
    def forward(s, b):
        bo = s.bert(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        return s.cls(bo.last_hidden_state[:,0,:]), bo.last_hidden_state[:,0,:]

# ============================================================
# Training & Evaluation
# ============================================================
def run_model(model, tr_l, vl_l, te_l, lr=1e-5, cw=3.0, ep=30, pat=6):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, cw], device=DEVICE))
    best_auc = 0; pc = 0; bs = None
    for e in range(1, ep+1):
        t0 = time.time()
        model.train()
        for b in tr_l:
            b = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            opt.zero_grad(); lo, _ = model(b); loss = crit(lo, b["labels"])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); la, pr = [], []
        with torch.no_grad():
            for b in vl_l:
                b = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                lo, _ = model(b); p = F.softmax(lo, -1)
                la.extend(b["labels"].cpu().numpy()); pr.extend(p[:,1].cpu().numpy())
        try: va = roc_auc_score(la, pr)
        except: va = 0
        if va > best_auc: best_auc = va; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}; pc = 0
        else: pc += 1
        print(f"    ep{e:2d}: val_AUROC={va:.4f} best={best_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if pc >= pat: break
    if bs: model.load_state_dict(bs)
    model = model.to(DEVICE); model.eval(); pa, la, pr = [], [], []
    with torch.no_grad():
        for b in te_l:
            b = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            lo, _ = model(b); p = F.softmax(lo, -1)
            pa.extend(lo.argmax(-1).cpu().numpy()); la.extend(b["labels"].cpu().numpy()); pr.extend(p[:,1].cpu().numpy())
    acc = accuracy_score(la, pa); rec = recall_score(la, pa, zero_division=0)
    f1 = f1_score(la, pa, zero_division=0)
    try: auc = roc_auc_score(la, pr)
    except: auc = 0
    mcc = matthews_corrcoef(la, pa); cm = confusion_matrix(la, pa)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    spec = tn/(tn+fp) if (tn+fp) > 0 else 0
    del model; torch.cuda.empty_cache()
    return {"acc":round(acc,4),"recall":round(rec,4),"specificity":round(spec,4),
            "f1":round(f1,4),"auc":round(auc,4),"mcc":round(mcc,4),"cm":cm.tolist()}

results = {}

# Load caches
print("Loading caches...", flush=True)
tr_cache = pickle.load(open(f"{CACHE_DIR}/train_dr0.pkl", 'rb'))
vl_cache = pickle.load(open(f"{CACHE_DIR}/val_dr0.pkl", 'rb'))
te_cache = pickle.load(open(f"{CACHE_DIR}/test_dr0.pkl", 'rb'))

trl = DataLoader(CachedDS(tr_cache), batch_size=8, shuffle=True, collate_fn=collate)
vll = DataLoader(CachedDS(vl_cache), batch_size=8, shuffle=False, collate_fn=collate)
tel = DataLoader(CachedDS(te_cache), batch_size=8, shuffle=False, collate_fn=collate)

# ============================================================
# 1. Main Comparison
# ============================================================
print(f"\n{'='*60}\n[1/6] Main Comparison\n{'='*60}", flush=True)

for name, ModelClass, kwargs in [
    ("BERT_Only", BERTOnly, {}),
    ("RGCN_Only", RGCNOnly, {}),
    ("Concat_Fusion", DualChannel, {"fusion": "concat"}),
    ("DualChannel_Ours", DualChannel, {"fusion": "crossattn"}),
]:
    print(f"\n  >> {name}", flush=True)
    m = ModelClass(MODEL_NAME, nc=2, **kwargs) if ModelClass != RGCNOnly else ModelClass(nc=2, **kwargs)
    r = run_model(m, trl, vll, tel, lr=1e-5, cw=3.0)
    results[name] = r
    print(f"  => AUROC={r['auc']} Rec={r['recall']} F1={r['f1']} MCC={r['mcc']}", flush=True)

# ============================================================
# 2. Ablation (fusion strategies + component removal)
# ============================================================
print(f"\n{'='*60}\n[2/6] Ablation Study\n{'='*60}", flush=True)

for name, fusion in [("AddFusion", "add"), ("GateFusion", "gate")]:
    print(f"\n  >> {name}", flush=True)
    m = DualChannel(MODEL_NAME, nc=2, fusion=fusion)
    r = run_model(m, trl, vll, tel, lr=1e-5, cw=3.0)
    results[name] = r
    print(f"  => AUROC={r['auc']} Rec={r['recall']} F1={r['f1']}", flush=True)

# w/o class weights (cw=1.0)
print(f"\n  >> NoClassWeights", flush=True)
m = DualChannel(MODEL_NAME, nc=2, fusion='crossattn')
r = run_model(m, trl, vll, tel, lr=1e-5, cw=1.0)
results["NoClassWeights"] = r
print(f"  => AUROC={r['auc']} Rec={r['recall']} F1={r['f1']}", flush=True)

# ============================================================
# 3. Hyperparameter Sensitivity
# ============================================================
print(f"\n{'='*60}\n[3/6] Hyperparameter Sensitivity\n{'='*60}", flush=True)

for lr_val in [1e-5, 3e-5]:
    for cw_val in [1.0, 2.0, 3.0]:
        name = f"LR{lr_val}_CW{cw_val}"
        print(f"\n  >> {name}", flush=True)
        m = DualChannel(MODEL_NAME, nc=2, fusion='crossattn')
        r = run_model(m, trl, vll, tel, lr=lr_val, cw=cw_val, ep=20, pat=5)
        results[name] = r
        print(f"  => AUROC={r['auc']} Rec={r['recall']} F1={r['f1']}", flush=True)

# ============================================================
# 4. K-BERT
# ============================================================
print(f"\n{'='*60}\n[4/6] K-BERT Baseline\n{'='*60}", flush=True)

tr_kb = pickle.load(open(f"{CACHE_DIR}/train_kbert.pkl", 'rb'))
vl_kb = pickle.load(open(f"{CACHE_DIR}/val_kbert.pkl", 'rb'))
te_kb = pickle.load(open(f"{CACHE_DIR}/test_kbert.pkl", 'rb'))
trl_kb = DataLoader(CachedDS(tr_kb, 'labels'), batch_size=8, shuffle=True, collate_fn=kb_collate)
vll_kb = DataLoader(CachedDS(vl_kb, 'labels'), batch_size=8, shuffle=False, collate_fn=kb_collate)
tel_kb = DataLoader(CachedDS(te_kb, 'labels'), batch_size=8, shuffle=False, collate_fn=kb_collate)

m = KBERTModel(MODEL_NAME, nc=2, dp=0.2)
r = run_model(m, trl_kb, vll_kb, tel_kb, lr=1e-5, cw=3.0)
results["KBERT"] = r
print(f"  => AUROC={r['auc']} Rec={r['recall']} F1={r['f1']}", flush=True)

# ============================================================
# 5. Robustness (entity drop)
# ============================================================
print(f"\n{'='*60}\n[5/6] Robustness (Entity Drop)\n{'='*60}", flush=True)

for dr in [10, 20, 30]:
    label = f"Drop{dr}pct"
    print(f"\n  >> {label}", flush=True)
    dr_cache = pickle.load(open(f"{CACHE_DIR}/train_dr{dr//10}_dr{dr}.pkl", 'rb'))
    trl_dr = DataLoader(CachedDS(dr_cache), batch_size=8, shuffle=True, collate_fn=collate)
    m = DualChannel(MODEL_NAME, nc=2, fusion='crossattn')
    r = run_model(m, trl_dr, vll, tel, lr=1e-5, cw=3.0, ep=20, pat=5)
    results[label] = r
    print(f"  => AUROC={r['auc']} Rec={r['recall']} F1={r['f1']}", flush=True)

# ============================================================
# 6. 5-Fold Cross-Validation
# ============================================================
print(f"\n{'='*60}\n[6/6] 5-Fold Cross-Validation\n{'='*60}", flush=True)

all_cache = pickle.load(open(f"{CACHE_DIR}/all_dr0.pkl", 'rb'))
labels_all = [s['label'].item() if isinstance(s['label'], torch.Tensor) else s['label'] for s in all_cache]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

fd_dc, fd_cat = [], []
for fold, (tri, tei) in enumerate(skf.split(range(len(all_cache)), labels_all)):
    print(f"\n  Fold {fold+1}/5:", flush=True)
    tf = [all_cache[i] for i in tri]; tef = [all_cache[i] for i in tei]
    nv = max(1, len(tf)//5); vf = tf[-nv:]; tft = tf[:-nv]
    trl2 = DataLoader(CachedDS(tft), batch_size=8, shuffle=True, collate_fn=collate)
    vll2 = DataLoader(CachedDS(vf), batch_size=8, shuffle=False, collate_fn=collate)
    tel2 = DataLoader(CachedDS(tef), batch_size=8, shuffle=False, collate_fn=collate)
    m = DualChannel(MODEL_NAME, nc=2, fusion='crossattn')
    r = run_model(m, trl2, vll2, tel2, lr=1e-5, cw=3.0, ep=20, pat=5)
    fd_dc.append(r); print(f"    DC: AUC={r['auc']} Rec={r['recall']} F1={r['f1']}", flush=True)
    m = DualChannel(MODEL_NAME, nc=2, fusion='concat')
    r = run_model(m, trl2, vll2, tel2, lr=1e-5, cw=3.0, ep=20, pat=5)
    fd_cat.append(r); print(f"    Cat: AUC={r['auc']} Rec={r['recall']} F1={r['f1']}", flush=True)

def ms(folds, k): v=[f[k] for f in folds]; return round(float(np.mean(v)),4), round(float(np.std(v)),4)
cv = {}
for k in ['acc','recall','specificity','f1','auc','mcc']:
    m1,s1 = ms(fd_dc, k); m2,s2 = ms(fd_cat, k)
    cv[k] = {'DC_mean':m1,'DC_std':s1,'Concat_mean':m2,'Concat_std':s2}
ta,pa = stats.ttest_rel([f['auc'] for f in fd_dc],[f['auc'] for f in fd_cat])
tf,pf = stats.ttest_rel([f['f1'] for f in fd_dc],[f['f1'] for f in fd_cat])
tr_,pr_ = stats.ttest_rel([f['recall'] for f in fd_dc],[f['recall'] for f in fd_cat])
cv['t_test'] = {'AUROC':{'t':round(float(ta),4),'p':round(float(pa),4)},
                'F1':{'t':round(float(tf),4),'p':round(float(pf),4)},
                'Recall':{'t':round(float(tr_),4),'p':round(float(pr_),4)}}
cv['per_fold_recall'] = {
    'DC': [round(f['recall'],4) for f in fd_dc],
    'Concat': [round(f['recall'],4) for f in fd_cat]
}
results['cv'] = cv

# Save
out_path = "D:/code/Hermes/ad-dualchannel-gcn/results/all_results.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

elapsed = time.time() - t_start
print(f"\n{'='*60}", flush=True)
print(f"ALL EXPERIMENTS COMPLETE (elapsed: {elapsed/60:.0f} min)", flush=True)
print(f"{'='*60}", flush=True)
for k, v in results.items():
    print(f"  {k:25s}: AUROC={v['auc']} Rec={v['recall']} F1={v['f1']} MCC={v['mcc']}", flush=True)
print(f"\nSaved to {out_path}", flush=True)
