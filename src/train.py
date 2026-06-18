"""
Training pipeline for Dual-Channel AD Detection Model
"""
import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)
from transformers import AutoTokenizer

from src.model import DualChannelADModel
from src.data import ClinicalNoteDataset, collate_fn
from configs.config import ModelConfig, TrainingConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(batch)
        logits = outputs["logits"]
        labels = batch["labels"]

        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        outputs = model(batch)
        logits = outputs["logits"]
        probs = outputs["probs"]
        labels = batch["labels"]

        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "specificity": _specificity(all_labels, all_preds),
        "mcc": matthews_corrcoef(all_labels, all_preds),
    }

    try:
        if all_probs.shape[1] == 2:
            metrics["auroc"] = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            metrics["auroc"] = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except ValueError:
        metrics["auroc"] = 0.0

    metrics["report"] = classification_report(all_labels, all_preds, zero_division=0)
    metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()

    return metrics


def _specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.0


def run_training(train_texts, train_labels, val_texts, val_labels,
                 model_config=None, train_config=None, model_name="dualchannel"):
    """Full training pipeline."""
    if model_config is None:
        model_config = ModelConfig()
    if train_config is None:
        train_config = TrainingConfig()

    set_seed(train_config.seed)
    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.bert_model_name)

    # Datasets
    train_dataset = ClinicalNoteDataset(
        train_texts, train_labels, tokenizer, train_config.max_seq_length)
    val_dataset = ClinicalNoteDataset(
        val_texts, val_labels, tokenizer, train_config.max_seq_length)

    train_loader = DataLoader(
        train_dataset, batch_size=train_config.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(
        val_dataset, batch_size=train_config.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Model
    model = DualChannelADModel(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Optimizer & loss
    optimizer = optim.AdamW(
        model.parameters(), lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training loop with early stopping
    best_auroc = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_metrics": []}

    os.makedirs(train_config.output_dir, exist_ok=True)

    for epoch in range(train_config.num_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_metrics"].append(val_metrics)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{train_config.num_epochs} "
              f"[{elapsed:.1f}s] "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val AUROC={val_metrics['auroc']:.4f} "
              f"F1={val_metrics['f1']:.4f} "
              f"Recall={val_metrics['recall']:.4f}")

        # Early stopping on AUROC
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_auroc": best_auroc,
                "val_metrics": val_metrics,
            }, os.path.join(train_config.output_dir, f"{model_name}_best.pt"))
            print(f"  -> Best model saved (AUROC={best_auroc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= train_config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save history
    with open(os.path.join(train_config.output_dir, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)

    return model, history
