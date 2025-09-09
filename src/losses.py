import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
	def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.reduction = reduction

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		log_probs = F.log_softmax(logits, dim=1)
		probs = log_probs.exp()
		pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
		log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
		loss = -self.alpha * (1 - pt) ** self.gamma * log_pt
		if self.reduction == "mean":
			return loss.mean()
		elif self.reduction == "sum":
			return loss.sum()
		return loss
