import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets, masks):
        raw_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
        masked_loss = raw_loss * masks
        valid_entries = masks.sum() + 1e-8
        return masked_loss.sum() / valid_entries

def calculate_masked_metrics(logits, targets, masks, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    valid_preds = preds * masks
    valid_targets = targets * masks
    
    true_positives = ((valid_preds == 1.0) & (valid_targets == 1.0) & (masks == 1.0)).sum(dim=0)
    false_positives = ((valid_preds == 1.0) & (valid_targets == 0.0) & (masks == 1.0)).sum(dim=0)
    false_negatives = ((valid_preds == 0.0) & (valid_targets == 1.0) & (masks == 1.0)).sum(dim=0)
    
    epsilon = 1e-8
    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return {
        'precision': precision.mean().item(), 
        'recall': recall.mean().item(), 
        'f1': f1_score.mean().item()
    }
    
def rand_bbox(size, lam):
    """Generates a random bounding box for CutMix augmentation."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
    
    
    