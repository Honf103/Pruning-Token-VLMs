import torch
import torch.nn as nn
import torch.nn.functional as F


class CLSScorer(nn.Module):
    """
    Parameter-free CLIP CLS scorer.

    Both cls_embedding and patch_tokens are assumed to already live in the same
    frozen CLIP joint embedding space. Scoring is plain cosine similarity,
    following the FasterVLM intuition without an extra learned projector.
    """

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()

    def forward(
        self,
        cls_embedding: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        cls_embedding : [B, D_clip]
        patch_tokens  : [B, N, D_clip]

        returns S_cls : [B, N]
        """
        cls_q = F.normalize(cls_embedding, dim=-1)
        patches_norm = F.normalize(patch_tokens, dim=-1)
        s_cls = torch.einsum("bd,bnd->bn", cls_q, patches_norm)
        return s_cls
