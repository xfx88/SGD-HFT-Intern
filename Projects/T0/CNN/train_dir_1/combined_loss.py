# !pip install torchsort

import torch
import torch.nn as nn
import torchsort

def corrcoef(target, pred):
    pred_n = pred - pred.mean() + torch.rand(pred.shape).cuda() * 1e-12
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()

def spearman(
    target,
    pred,
    regularization="kl",
    regularization_strength=1.0,
    weight = (0.2, 0.3, 0.5)):

    corr = 0
    for i in range(pred.shape[-1]):
        pred_i = pred[:, i : i + 1].T
        target_i = target[:, i : i + 1].T
        pred_i = torchsort.soft_rank(
                 pred_i,
                 regularization=regularization,
                 regularization_strength=regularization_strength,)

        corr += weight[i] * corrcoef(target_i, pred_i / pred_i.shape[-1])

    return 1 - corr

class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(self, pred, target):
        spearman_loss = spearman(pred = pred, target = target)
        mse = self.mseloss(pred, target)
        return 1e6 * spearman_loss * mse