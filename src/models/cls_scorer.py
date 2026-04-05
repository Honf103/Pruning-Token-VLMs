import torch
import torch.nn as nn
import torch.nn.functional as F


class CLSScorer(nn.Module):
    """
    Visual saliency score via CLIP CLS–patch cosine similarity.

    Computes how similar each patch token is to the CLIP CLS embedding,
    giving a pure image-side importance signal independent of the question.

    Pipeline:
        cls_embedding  → Linear(vision_dim → llm_dim) → L2-norm  → cls_q [B, D]
        patch_tokens   → L2-norm                                   → [B, N, D]
        S_cls = cls_q · patch_tokens^T                             → [B, N]

    This captures CLIP's inherent visual saliency — the CLS token encodes
    the global image representation and its cosine similarity with each patch
    reflects how much each region contributes to the overall image semantics.
    Follows the intuition of FasterVLM (Liang et al. 2022).
    """

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        # Project CLS from CLIP vision space (1024-d) to LLM token space (4096-d)
        # so the comparison happens in the same space as the projected patch tokens.
        self.cls_proj = nn.Linear(vision_dim, llm_dim, bias=False)

    def forward(
        self,
        cls_embedding: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        cls_embedding : [B, vision_dim]  — raw CLIP CLS token
        patch_tokens  : [B, N, llm_dim]  — projected visual tokens z

        returns S_cls : [B, N]  — cosine similarity of each patch with CLS
        """
        cls_q = self.cls_proj(cls_embedding)              # [B, llm_dim]
        cls_q = F.normalize(cls_q, dim=-1)                # unit norm  [B, D]
        patches_norm = F.normalize(patch_tokens, dim=-1)  # [B, N, D]
        s_cls = torch.einsum("bd,bnd->bn", cls_q, patches_norm)  # [B, N]
        return s_cls
