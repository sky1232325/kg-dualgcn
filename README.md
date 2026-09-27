# Leakage-Controlled Text--Graph Fusion on Heterogeneous Chinese Medical Text

Code, data-audit artefacts, and manuscript source for:

> **When Weak Labels Become the Task: Leakage-Controlled Evaluation of Text--Graph Fusion on Heterogeneous Chinese Medical Text**

The study turns an apparently clinical text--graph prediction problem into an explicit audit of label provenance,
leakage control, fusion behaviour, and explanation faithfulness. It is **not** an estimate of clinical diagnostic or
screening performance, and the repository makes no clinical claim.

## What this repository contains

```
src/                          Model implementation
  model.py                    Dual-channel text--graph model
  cross_attention.py          Bidirectional cross-attention fusion
  rgcn_layer.py               R-GCN layer over the source graph
  data.py                     Dataset and DataLoader
  train.py                    Training pipeline
  explainer.py                Integrated Gradients and GNNExplainer
paper/                        LaTeX source and figures
run_all_gpu_experiments.py    Full experiment suite
run_ml_baselines.py           Baseline classifiers (including TextCNN)
requirements.txt
```

## Key findings

- The original binary labels are reproduced **exactly** (accuracy 1.000) by a deterministic keyword rule,
  so they are not independent clinician annotations.
- 36 exact duplicate records were identified; the source graph's 35,867 edges are co-occurrence edges
  rather than clinician-defined semantic relations.
- On the resulting 3,642-record proxy-label benchmark, cross-attention reaches **AUROC 0.9559**
  (95% bootstrap CI [0.9392, 0.9701]) and exceeds the matched frozen-text comparator by
  **0.0174 AUROC** (95% CI [0.0058, 0.0295], paired bootstrap p = 0.004).
- An independent TextCNN baseline reaches **0.9607 +/- 0.0030 AUROC**, so the dual-channel model is
  **not** the strongest evaluated classifier. This is reported as a constraint on the ranking.
- Attribution deletion and graph occlusion are quantitatively faithful under the proxy task, but the
  inspected attributions are **not** evidence of clinical reasoning.

## External validation status

```
blocked: missing authorized gold-label data
```

No authorized clinician-confirmed external corpus is available locally, and no MIMIC, ADNI, or equivalent
gold-standard evaluation was performed. This is reported as the single remaining obstacle between this audit
and a clinical study, rather than left implicit.

## Source corpus

The source corpus is the publicly available medical NER resource hosted at
`https://huggingface.co/datasets/ShelterW/chinese_medical_ner`, used under the original provider's data-use terms.
The raw source records are **not** rehosted here; the sanitised proxy-label benchmark and leakage-audit artefacts are.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.8+
- Transformers 4.30+

## Quick start

```bash
pip install -r requirements.txt
python run_all_gpu_experiments.py
```

## Citation

```bibtex
@article{liu2026weaklabels,
  title={When Weak Labels Become the Task: Leakage-Controlled Evaluation of Text--Graph Fusion on Heterogeneous Chinese Medical Text},
  author={Liu, Chun and Yang, Rui and Guo, Haitao},
  year={2026}
}
```

## Licence

MIT. See `LICENSE`.
