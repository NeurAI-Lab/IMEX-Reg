import torch
import torch.nn.functional as F
import torch.nn as nn


class ECR:

    def __init__(self):
        pass

    def update(self, logits, projections):
        bsz = logits.shape[0]
        G_s = torch.mm(logits, torch.t(logits))
        G_t = torch.mm(projections, torch.t(projections))
        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss
