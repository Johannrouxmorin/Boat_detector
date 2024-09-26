import torch
import torch.nn as nn
import torch.nn.functional as F

def iou_score(y_pred, y_true, smooth=1):
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        return dice + bce

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred) 
        intersection = (y_pred * y_true).sum()
        total = (y_pred + y_true).sum()
        union = total - intersection
        
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        return 1 - jaccard

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        pos_pairs = torch.pow(y_pred_flat - y_true_flat, 2)
        neg_pairs = torch.pow(torch.clamp(self.margin - (y_pred_flat - y_true_flat), min=0.0), 2)
        
        loss = 0.5 * (y_true_flat * pos_pairs + (1 - y_true_flat) * neg_pairs).mean()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_true = y_true.float()

        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss