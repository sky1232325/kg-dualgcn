# KG-DualGCN

Knowledge-Guided Dual-Channel Graph Convolutional Networks for Explainable Alzheimer's Disease Screening in Clinical Notes.

## Overview

KG-DualGCN is a dual-channel architecture that encodes clinical text via BioClinicalBERT and medical entity relationships via R-GCN, fused through bidirectional cross-attention. Dual-view explanation modules (Integrated Gradients and GNNExplainer) provide interpretable diagnostic evidence.

## Project Structure

```
├── src/                    # Model implementation
│   ├── model.py           # KG-DualGCN model
│   ├── cross_attention.py # Bidirectional cross-attention
│   ├── rgcn_layer.py      # R-GCN layer
│   ├── data.py            # Dataset & DataLoader
│   ├── train.py           # Training pipeline
│   └── explainer.py       # IG & GNNExplainer
├── paper/                  # LaTeX source & figures
├── run_all_gpu_experiments.py  # Complete experiment suite
├── run_ml_baselines.py         # Traditional ML baselines
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python run_all_gpu_experiments.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.8+
- Transformers 4.30+

## Citation

If you use this code, please cite:

```bibtex
@article{liu2026kgdualgcn,
  title={Knowledge-Guided Dual-Channel Graph Convolutional Networks for Explainable Alzheimer's Disease Screening in Clinical Notes},
  author={Liu, Chun and Yang, Rui and Guo, Haitao},
  journal={Applied Intelligence},
  year={2026}
}
```

## License

MIT License
