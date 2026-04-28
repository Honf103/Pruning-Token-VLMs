# Prompt for Implementing Knowledge Distillation (KD) Upgrade in `Honf103/Pruning-Token-VLMs`

Use the following prompt in VS Code / Visual Studio with your coding LLM agent to implement the KD-focused training upgrade.

---

You are an expert ML engineer working on a Vision-Language Model token pruning research project.
The codebase is located in the `src/` directory. Your task is to significantly upgrade
the training pipeline by implementing proper Knowledge Distillation (KD) and freezing
the LLM + Projector backbone for fair comparison with existing methods.

## CORE OBJECTIVE

Transform Stage 2 training so that:
- FROZEN: vision_encoder, text_encoder, projector, LLM (all ~7B params)
- TRAINABLE: CLSScorer, TextImportanceMLP, InstructionAwareScorer, ScoreFusion (~10-12M params)
- REPLACE the weak `align_loss` (L2 norm proxy) with proper LLM cross-attention distillation
- ADD question-conditioned alpha to ScoreFusion (instead of a single learnable scalar)
- ADD a dedicated KD loss for alpha supervision

## FILE-BY-FILE CHANGES

─────────────────────────────────────────────────
### FILE 1: `src/models/score_fusion.py`
─────────────────────────────────────────────────

REPLACE the entire file with the following logic:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreFusion(nn.Module):
    """
    Question-conditioned score fusion.

    Instead of a single global learnable alpha, predict alpha per sample
    from the CLIP text CLS token. This allows the model to learn:
      - spatial/counting questions → alpha→1 (instruction-aware dominates)
      - simple recognition questions → alpha→0 (CLS saliency suffices)

    S = alpha(q) * zscore(S_ia) + (1 - alpha(q)) * zscore(S_cls)

    alpha_head is a small MLP: text_cls [B, text_dim] → alpha [B, 1] ∈ (0,1)

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
            # MLP: text_cls → alpha ∈ (0,1)
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
        text_cls_token : [B, text_dim] — first token of CLIP text encoder output
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
```

─────────────────────────────────────────────────
### FILE 2: `src/models/pruning_vlm.py`
─────────────────────────────────────────────────

Make the following targeted changes:

#### 2a. In `__init__`, update ScoreFusion instantiation

FIND this block:
```python
        self.score_fusion = ScoreFusion(
            alpha=alpha,
            learnable=learnable_alpha,
        )
```

REPLACE WITH:
```python
        self.score_fusion = ScoreFusion(
            alpha=alpha,
            learnable=learnable_alpha,
            question_conditioned=question_conditioned_alpha,
            text_dim=text_dim,
            hidden_dim=64,
        )
```

#### 2b. Add new `__init__` parameters

In the `__init__` signature, ADD these parameters after `use_merging: bool = False`:
```python
        question_conditioned_alpha: bool = False,
        text_dim: int = 768,
        attn_distill_layers: list = None,   # e.g. [8, 16, 23] — LLM layers for KD
```

And in the body of `__init__`, ADD after `self.use_merging = bool(use_merging)`:
```python
        self.question_conditioned_alpha = bool(question_conditioned_alpha)
        self.attn_distill_layers = attn_distill_layers or [8, 16, 23]
```

#### 2c. Update `forward` — ScoreFusion call

FIND:
```python
        fused_scores = self.score_fusion(s_ia, s_cls)  # [B, N]
```

REPLACE WITH:
```python
        # For question-conditioned alpha, pass text CLS token (index 0)
        text_cls = text_tokens[:, 0, :] if self.question_conditioned_alpha else None
        fused_scores, predicted_alpha = self.score_fusion(s_ia, s_cls, text_cls_token=text_cls)  # [B, N], [B,1] or scalar
```

#### 2d. Update return dict

FIND:
```python
        out = {
            "logits": logits,
            "loss": loss,
            "fused_scores": fused_scores,
```

REPLACE WITH:
```python
        out = {
            "logits": logits,
            "loss": loss,
            "fused_scores": fused_scores,
            "predicted_alpha": predicted_alpha,
```

#### 2e. REPLACE `align_loss` computation with proper LLM cross-attention KD

FIND the entire block starting with:
```python
            # ── Align: KL(S_ia || hidden-state importance at align_layer) ─────
            if compute_align_loss and s_ia is not None:
```
...and ending with the closing of that if block.

REPLACE WITH:
```python
            # ── Align: KL(S_ia || LLM cross-attention to visual tokens) ────────
            # Uses actual LLM attention weights (text→visual) as teacher signal.
            # This is strictly better than L2-norm of hidden states because it
            # directly measures which visual tokens the LLM attends to.
            if compute_align_loss and s_ia is not None:
                t_attentions = (
                    teacher_out.get("attentions") if isinstance(teacher_out, dict)
                    else getattr(teacher_out, "attentions", None)
                )
                if t_attentions is not None and len(t_attentions) > 0:
                    # Collect attention to visual tokens from target layers
                    # visual tokens are the first N_vis tokens in the LLM sequence
                    layer_importances = []
                    for layer_idx in self.attn_distill_layers:
                        if layer_idx >= len(t_attentions):
                            continue
                        attn = t_attentions[layer_idx]  # [B, num_heads, seq_len, seq_len]
                        # text tokens attend to visual prefix: attn[:, :, N_vis:, :N_vis]
                        seq_len = attn.size(2)
                        if seq_len > N_vis:
                            # [B, num_heads, L_text, N_vis] → mean over heads and text positions
                            attn_to_vis = attn[:, :, N_vis:, :N_vis]        # [B, h, L, N]
                            importance = attn_to_vis.mean(dim=(1, 2))        # [B, N]
                            layer_importances.append(importance)

                    if layer_importances:
                        # Average importance across selected layers
                        teacher_importance = torch.stack(layer_importances, dim=0).mean(0)  # [B, N]
                        # Clamp to avoid degenerate softmax
                        teacher_importance = teacher_importance.clamp(min=0.0)

                        align_loss = F.kl_div(
                            F.log_softmax(s_ia, dim=-1),
                            F.softmax(teacher_importance.detach() * 10.0, dim=-1),  # sharpen
                            reduction="batchmean",
                        )
```

#### 2f. Update the teacher forward call to request attentions

FIND:
```python
                with torch.no_grad():
                    teacher_out = self.llm(
                        projected_visual_tokens=z,
                        input_ids=_llm_in,
                        attention_mask=_llm_attn,
                        visual_attention_mask=None,
                        labels=None,
                        output_hidden_states=compute_align_loss,
                    )
```

REPLACE WITH:
```python
                with torch.no_grad():
                    teacher_out = self.llm(
                        projected_visual_tokens=z,
                        input_ids=_llm_in,
                        attention_mask=_llm_attn,
                        visual_attention_mask=None,
                        labels=None,
                        output_hidden_states=False,
                        output_attentions=compute_align_loss,
                    )
```

#### 2g. Add `alpha_kd_loss` to the output dict

ADD to the `out` dict:
```python
            "alpha_kd_loss": None,
```

─────────────────────────────────────────────────
### FILE 3: `src/models/hf_backbones.py`
─────────────────────────────────────────────────

Find the `LLaVALM.forward` method.
ADD `output_attentions: bool = False` to its signature.
Pass it through to the underlying HF model call:
```python
outputs = self.model(
    ...
    output_attentions=output_attentions,
    ...
)
```
And return `attentions` in the output dict:
```python
return {
    "logits": ...,
    "loss": ...,
    "hidden_states": ...,
    "attentions": getattr(outputs, "attentions", None),
}
```

─────────────────────────────────────────────────
### FILE 4: `src/train.py`
─────────────────────────────────────────────────

#### 4a. Add new function: `compute_alpha_supervision`

ADD this new function after `compute_budget_loss`:

```python
@torch.no_grad()
def compute_alpha_supervision(
    model: "PruningVLM",
    images: torch.Tensor,
    clip_input_ids: torch.Tensor,
    clip_attention_mask: torch.Tensor,
    llm_input_ids: torch.Tensor,
    llm_attention_mask: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    Compute alpha supervision target using teacher LM loss.

    For each batch:
      loss_ia  = LM loss when pruning with alpha=1.0 (instruction-aware only)
      loss_cls = LM loss when pruning with alpha=0.0 (CLS saliency only)
      alpha*   = sigmoid((loss_cls - loss_ia) / temperature)
    """
    model.score_fusion.alpha_override = 1.0
    out_ia = model(
        images=images,
        input_ids=clip_input_ids,
        attention_mask=clip_attention_mask,
        llm_input_ids=llm_input_ids,
        llm_attention_mask=llm_attention_mask,
        labels=labels,
        use_hard_pruning=True,
        train_pruning_mode="structural",
    )
    loss_ia = out_ia.get("loss", None)

    model.score_fusion.alpha_override = 0.0
    out_cls = model(
        images=images,
        input_ids=clip_input_ids,
        attention_mask=clip_attention_mask,
        llm_input_ids=llm_input_ids,
        llm_attention_mask=llm_attention_mask,
        labels=labels,
        use_hard_pruning=True,
        train_pruning_mode="structural",
    )
    loss_cls = out_cls.get("loss", None)

    model.score_fusion.alpha_override = None

    if loss_ia is None or loss_cls is None:
        return None
    if not (torch.isfinite(loss_ia) and torch.isfinite(loss_cls)):
        return None

    alpha_target = torch.sigmoid((loss_cls - loss_ia) / temperature)
    return alpha_target.detach().expand(images.size(0))
```

#### 4b. Update `train_stage2` freeze strategy

In `train_stage2`, replace the current Stage 2 freeze/unfreeze logic with:

```python
    freeze_module(model.vision_encoder)
    freeze_module(model.text_encoder)
    freeze_module(model.projector)
    for p in model.llm.model.parameters():
        p.requires_grad = False

    unfreeze_module(model.cls_scorer)
    unfreeze_module(model.text_importance)
    unfreeze_module(model.instruction_aware)
    unfreeze_module(model.score_fusion)

    print("[Stage2] Backbone FROZEN. Training scorer only (~12M params).")
    print_trainable_parameters(model)
```

Remove the `freeze_llm=False` / LoRA branch from the experiment path.

#### 4c. Add new `train_stage2` arguments

ADD these arguments to the function signature:
```python
    lambda_align: float = 0.3,
    lambda_alpha_kd: float = 0.1,
    alpha_kd_every_n_steps: int = 10,
    question_conditioned_alpha: bool = True,
    attn_distill_layers: list = None,
```

#### 4d. Update model construction in `train_stage2`

ADD these args to `PruningVLM(...)`:
```python
        question_conditioned_alpha=question_conditioned_alpha,
        text_dim=768,
        attn_distill_layers=attn_distill_layers or [8, 16, 23],
```

#### 4e. Enable `compute_align_loss` in forward call

Update the training-step forward call so it includes:
```python
                    compute_distill_loss=(current_lambda_distill_step > 0),
                    compute_align_loss=(lambda_align > 0),
                    align_layer=(attn_distill_layers[0] if attn_distill_layers else 8),
```

#### 4f. Update `full_loss`

Replace the current `full_loss` block with:

```python
                align_loss = outputs.get("align_loss")
                if align_loss is None:
                    align_loss = fused_scores.new_tensor(0.0)

                alpha_kd_loss = fused_scores.new_tensor(0.0)
                if (
                    question_conditioned_alpha
                    and lambda_alpha_kd > 0
                    and _global_step % alpha_kd_every_n_steps == 0
                ):
                    with torch.no_grad():
                        alpha_target = compute_alpha_supervision(
                            model=model,
                            images=images,
                            clip_input_ids=clip_input_ids,
                            clip_attention_mask=clip_attention_mask,
                            llm_input_ids=llm_input_ids,
                            llm_attention_mask=llm_attention_mask,
                            labels=labels,
                            temperature=0.5,
                        )
                    if alpha_target is not None:
                        predicted_alpha = outputs.get("predicted_alpha")
                        if predicted_alpha is not None:
                            pa = predicted_alpha.squeeze()
                            alpha_kd_loss = F.mse_loss(pa, alpha_target.to(pa.device))

                full_loss = (
                    lm_loss
                    + current_lambda_budget_step * budget_loss
                    + current_lambda_distill_step * distill_loss
                    + lambda_align * align_loss
                    + lambda_alpha_kd * alpha_kd_loss
                )
```

#### 4g. Update logging

In the step logging print block, ADD these terms:
```python
f"AlignLoss {align_loss.item():.4f} | "
f"AlphaKD {alpha_kd_loss.item():.4f} | "
f"Alpha(mean) {outputs['predicted_alpha'].mean().item():.4f} | "
```

#### 4h. Update the bottom-level `train_stage2(...)` call

Use a fair-comparison KD-focused setup:

```python
    train_stage2(
        stage1_checkpoint=STAGE1_DIR,
        llava15_init=True,
        llava15_model_name="llava-hf/llava-1.5-7b-hf",
        num_epochs=stage2_num_epochs,
        batch_size=2,
        num_workers=4,
        grad_accum_steps=8,
        max_grad_norm=0.5,
        adapter_lr=2e-5,
        weight_decay=0.01,
        keep_ratio=0.333,
        keep_ratio_start=1.0,
        soft_stage_ratio=0.33,
        ste_stage_ratio=0.33,
        lambda_budget=0.03,
        lambda_distill=1.0,
        lambda_align=0.5,
        lambda_alpha_kd=0.2,
        alpha_kd_every_n_steps=20,
        question_conditioned_alpha=True,
        attn_distill_layers=[8, 16, 23],
        start_temp=1.5,
        end_temp=0.5,
        max_samples=stage2_max_samples,
        multi_ratio_enabled=True,
        multi_ratio_values=[0.056, 0.111, 0.223, 0.4, 0.5, 1.0],
        multi_ratio_probs=[1.5, 1.5, 1.4, 1.2, 1.0, 1.0],
        use_ratio_adaptive_loss=True,
        dynamic_budget_enabled=False,
        projector_unfreeze_threshold=0.0,
        enable_tf32=True,
        resume_from=_stage2_resume_from,
        projector_hidden_dim=2048,
        ia_hidden_dim=512,
        best_checkpoint_name=BEST_STAGE2_DIR_NAME,
        latest_checkpoint_name=LATEST_STAGE2_DIR_NAME,
    )
```

─────────────────────────────────────────────────
## IMPORTANT CONSTRAINTS
─────────────────────────────────────────────────

Do NOT modify:
- `src/models/token_pruner.py`
- `src/models/cls_scorer.py`
- `src/models/text_importance.py`
- `src/models/instruction_aware.py`
- Stage 1 training logic
- Evaluation scripts (`eval.py`, `test_mme.py`, `test_pope.py`, `infer.py`)

Only make the changes described above.

─────────────────────────────────────────────────
## AFTER IMPLEMENTATION: VERIFY
─────────────────────────────────────────────────

Please also run a quick sanity pass in code (do not execute training):

1. Ensure `PruningVLM(question_conditioned_alpha=True)` initializes without signature errors.
2. Ensure `LLaVALM.forward(..., output_attentions=True)` returns `attentions`.
3. Ensure `ScoreFusion.forward(...)` returns `(fused_scores, alpha)`.
4. Ensure Stage 2 freezes projector + LLM and only scorer modules remain trainable.
5. Print expected trainable parameter count in comments or logs if easy.

When finished, summarize exactly which files were changed and what each change does.
