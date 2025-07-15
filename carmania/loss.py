import torch
import torch.nn as nn
import torch.nn.functional as F





class TMLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, true_probs):
        probs = F.softmax(logits, dim=2)
        p1, p2 = probs[:, :-1, :], probs[:, 1:, :]
        pred_bigram = torch.einsum('bti,btj->bij', p1, p2)

        pred_bigram = pred_bigram[:, :-1, :-1]
        row_sums = pred_bigram.sum(dim=-1, keepdim=True).clamp_min(1)
        pred_bigram = pred_bigram / row_sums

        pred_bigram += self.epsilon
        true_probs += self.epsilon

        kl = torch.sum(true_probs * (torch.log(true_probs) - torch.log(pred_bigram)), dim=(-2, -1))
        return kl.mean()
