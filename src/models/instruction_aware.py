import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstructionAwareScorer(nn.Module):
    """
    Parameter-free text-to-vision cross-attention in frozen CLIP space.

    text_tokens and visual_tokens are assumed to already be in the same CLIP
    joint embedding space. The only learned question weighting comes from beta;
    the attention map itself is just scaled dot-product similarity.
    """

    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        text_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        beta: torch.Tensor,
        text_attention_mask: torch.Tensor = None,
        return_logits: bool = False,
    ):
        """
        text_tokens        : [B, L, D_clip]
        visual_tokens      : [B, N, D_clip]
        beta               : [B, L]
        text_attention_mask: [B, L]  1=valid, 0=pad
        """
        text_norm = F.normalize(text_tokens, dim=-1)
        visual_norm = F.normalize(visual_tokens, dim=-1)

        scale = 1.0 / (math.sqrt(text_norm.size(-1)) * max(float(self.temperature), 1e-6))
        attn_logits = torch.matmul(text_norm, visual_norm.transpose(-2, -1)) * scale

        if text_attention_mask is not None:
            attn_logits = attn_logits.masked_fill(
                text_attention_mask.unsqueeze(-1) == 0,
                float("-inf"),
            )

        A = F.softmax(attn_logits, dim=-1)

        if beta is not None:
            if text_attention_mask is not None:
                A = A * text_attention_mask.unsqueeze(-1).to(A.dtype)
                beta = beta * text_attention_mask.to(beta.dtype)
            beta = beta / (beta.sum(dim=-1, keepdim=True) + 1e-8)
            s_ia = torch.einsum("bl,bln->bn", beta, A)
        else:
            if text_attention_mask is not None:
                mask = text_attention_mask.unsqueeze(-1).to(A.dtype)
                s_ia = (A * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-8)
            else:
                s_ia = A.mean(dim=1)

        if return_logits:
            return s_ia, A, attn_logits

        return s_ia, A
