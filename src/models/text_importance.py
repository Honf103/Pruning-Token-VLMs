import torch
import torch.nn as nn
import torch.nn.functional as F


class TextImportanceMLP(nn.Module):
    """
    beta = softmax(MLP_text(T))
    T: [B, L, D]
    beta: [B, L]
    """
    def __init__(self, dim=256, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, text_tokens: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        text_tokens: [B, L, D]
        attention_mask: [B, L] with 1 for valid, 0 for pad
        return beta: [B, L]
        """
        logits = self.mlp(text_tokens).squeeze(-1)  # [B, L]

        if attention_mask is not None:
            logits = logits.masked_fill(attention_mask == 0, float("-inf"))

        beta = F.softmax(logits, dim=-1)
        return beta
