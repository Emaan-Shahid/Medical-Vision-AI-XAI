import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             recall_score, roc_auc_score)


@torch.no_grad()
def predict(model, loader, device):
    """Return the AI's probabilities (one row per picture) and the true labels."""
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(y_true, probs, classes):
    """All the scores for one set of pictures."""
    y_pred = probs.argmax(1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None,
                                    labels=list(range(len(classes))), zero_division=0)
    out = {
        'n': len(y_true),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': prec,
        'recall_macro': rec,
        'f1_macro': f1,
        'roc_auc_macro': roc_auc_score(y_true, probs, multi_class='ovr', average='macro'),
    }
    for c, r in zip(classes, per_class_recall):
        out[f'recall_{c}'] = r
    return out
