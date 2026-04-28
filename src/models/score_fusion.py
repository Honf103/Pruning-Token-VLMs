import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreFusion(nn.Module):
    """
    Question-conditioned score fusion.

    Instead of a single global learnable alpha, predict alpha per sample
    from the CLIP text CLS token. This allows the model to learn:
      - spatial/counting questions -> alpha->1 (instruction-aware dominates)
      - simple recognition questions -> alpha->0 (CLS saliency suffices)

    S = alpha(q) * zscore(S_ia) + (1 - alpha(q)) * zscore(S_cls)

    alpha_head is a small MLP: text_cls [B, text_dim] -> alpha [B, 1] in (0,1)

    KD supervision for alpha:
      alpha_target is computed externally (in train.py) using teacher LM loss
      comparison between S_ia-only and S_cls-only pruning.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        learnable: bool = False,
        question_conditioned: bool = False,
        text_dim: int = 768,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.question_conditioned = question_conditioned
        self.alpha_override = None

        if question_conditioned:
            # MLP: text_cls -> alpha in (0,1)
            self.alpha_head = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
            # Register a non-trainable buffer just for logging
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        elif learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    @staticmethod
    def _zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + eps)

    def forward(
        self,
        s_ia: torch.Tensor,
        s_cls: torch.Tensor,
        text_cls_token: torch.Tensor = None,
    ):
        """
        s_ia           : [B, N]
        s_cls          : [B, N]
        text_cls_token : [B, text_dim] - first token of CLIP text encoder output
                         Required when question_conditioned=True.

        Returns:
            s     : [B, N]  fused score
            alpha : [B, 1] or scalar  (for logging and KD loss)
        """
        # Allow external override for supervision computation
        if self.alpha_override is not None:
            override_val = float(self.alpha_override)
            alpha = torch.full(
                (s_ia.size(0), 1),
                override_val,
                device=s_ia.device,
                dtype=s_ia.dtype,
            )
            s = alpha * self._zscore(s_ia) + (1.0 - alpha) * self._zscore(s_cls)
            return s, alpha

        if self.question_conditioned:
            if text_cls_token is None:
                raise ValueError(
                    "text_cls_token is required when question_conditioned=True. "
                    "Pass text_tokens[:, 0, :] (CLIP text CLS embedding)."
                )
            alpha = self.alpha_head(text_cls_token.float())  # [B, 1]
        else:
            alpha = torch.clamp(self.alpha, 0.0, 1.0)

        s = alpha * self._zscore(s_ia) + (1.0 - alpha) * self._zscore(s_cls)
        return s, alpha
