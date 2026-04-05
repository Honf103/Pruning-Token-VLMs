import torch
import torch.nn as nn


class ScoreFusion(nn.Module):
    """
    Fuse two token importance scores:
        S = alpha * norm(S_ia) + (1 - alpha) * norm(S_cls)

    normalization: per-sample z-score so the two signals live on the same
    scale before fusion, preventing one branch from dominating the other.
    """

    def __init__(self, alpha: float = 0.5, learnable: bool = False):
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    @staticmethod
    def _zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Per-sample z-score normalization over the token dimension."""
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + eps)

    def forward(self, s_ia: torch.Tensor, s_cls: torch.Tensor) -> torch.Tensor:
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        s = alpha * self._zscore(s_ia) + (1.0 - alpha) * self._zscore(s_cls)
        return s