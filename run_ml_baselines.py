"""
Traditional ML baselines for AD detection: TF-IDF + LR, Word2Vec + XGBoost.
CPU only, no GPU needed.
"""
import json, random, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from gensim.models import Word2Vec
from xgboost import XGBClassifier

SEED = 42
random.seed(SEED); np.random.seed(SEED)

DATA_DIR = "D:/code/Hermes/ad-dualchannel-gcn/data/processed"

def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "specificity": round(spec, 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc": round(auc, 4),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
        "confusion_matrix": cm.tolist()
    }

# Load data
train_data = load_json(f"{DATA_DIR}/train.json")
val_data = load_json(f"{DATA_DIR}/val.json")
test_data = load_json(f"{DATA_DIR}/test.json")

train_texts = [d['text'] for d in train_data]
train_labels = [d['label'] for d in train_data]
val_texts = [d['text'] for d in val_data]
val_labels = [d['label'] for d in val_data]
test_texts = [d['text'] for d in test_data]
test_labels = [d['label'] for d in test_data]

# Combine train+val for ML baselines (standard practice)
all_train_texts = train_texts + val_texts
all_train_labels = train_labels + val_labels

print(f"Data: train={len(all_train_texts)}, test={len(test_labels)}")
print(f"  Train AD={sum(all_train_labels)}, NC={len(all_train_labels)-sum(all_train_labels)}")
print(f"  Test AD={sum(test_labels)}, NC={len(test_labels)-sum(test_labels)}")

results = {}

# ============================================================
# 1. TF-IDF + Logistic Regression
# ============================================================
print(f"\n{'='*60}")
print("[1/2] TF-IDF + Logistic Regression")
print(f"{'='*60}")
t0 = time.time()

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(all_train_texts)
X_test_tfidf = tfidf.transform(test_texts)

lr = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=SEED)
lr.fit(X_train_tfidf, all_train_labels)

y_pred = lr.predict(X_test_tfidf)
y_prob = lr.predict_proba(X_test_tfidf)[:, 1]

metrics = compute_metrics(test_labels, y_pred, y_prob)
results['TFIDF_LR'] = metrics
print(f"  AUROC={metrics['auc']} Recall={metrics['recall']} F1={metrics['f1']} Acc={metrics['accuracy']} ({time.time()-t0:.1f}s)")

# ============================================================
# 2. Word2Vec + XGBoost
# ============================================================
print(f"\n{'='*60}")
print("[2/2] Word2Vec + XGBoost")
print(f"{'='*60}")
t0 = time.time()

# Tokenize for Word2Vec
train_tokenized = [text.split() for text in all_train_texts]
test_tokenized = [text.split() for text in test_texts]

w2v = Word2Vec(sentences=train_tokenized, vector_size=128, window=5, min_count=2, workers=4, seed=SEED, epochs=20)

def text_to_vec(tokens, model, dim=128):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if vecs:
        return np.mean(vecs, axis=0)
    return np.zeros(dim)

X_train_w2v = np.array([text_to_vec(t, w2v) for t in train_tokenized])
X_test_w2v = np.array([text_to_vec(t, w2v) for t in test_tokenized])

xgb = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=sum(1-l for l in all_train_labels)/sum(all_train_labels),
    random_state=SEED, eval_metric='auc', use_label_encoder=False,
    tree_method='hist'  # CPU fast method
)
xgb.fit(X_train_w2v, all_train_labels)

y_pred = xgb.predict(X_test_w2v)
y_prob = xgb.predict_proba(X_test_w2v)[:, 1]

metrics = compute_metrics(test_labels, y_pred, y_prob)
results['W2V_XGBoost'] = metrics
print(f"  AUROC={metrics['auc']} Recall={metrics['recall']} F1={metrics['f1']} Acc={metrics['accuracy']} ({time.time()-t0:.1f}s)")

# ============================================================
# Save
# ============================================================
out_path = "D:/code/Hermes/ad-dualchannel-gcn/results/ml_baselines.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("ML BASELINE RESULTS")
print(f"{'='*60}")
for k, v in results.items():
    print(f"  {k:20s}: AUROC={v['auc']} Recall={v['recall']} F1={v['f1']} MCC={v['mcc']}")
print(f"\nSaved to {out_path}")
