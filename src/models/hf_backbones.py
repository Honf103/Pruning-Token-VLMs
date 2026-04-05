import torch
import torch.nn as nn


def _get_hf_transformers():
    try:
        import transformers
    except ImportError as e:
        raise ImportError(
            "transformers is required for HF backbones. "
            "Install it via `pip install transformers`."
        ) from e
    return transformers


class CLIPVisionEncoder(nn.Module):
    """
    Wrapper around HuggingFace CLIP vision encoder.

    Returns patch-level visual embeddings
    (including the [CLS] token unless removed).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        normalize: bool = True,
        remove_cls_token: bool = False,
    ):
        super().__init__()
        transformers = _get_hf_transformers()

        self.model = transformers.CLIPVisionModel.from_pretrained(model_name)
        self.config = self.model.config
        self.normalize = normalize
        self.remove_cls_token = remove_cls_token

        self.register_buffer(
            "_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor):
        """
        Returns (cls_embedding, patch_tokens):
            cls_embedding : [B, D]      — CLIP CLS token (raw vision space)
            patch_tokens  : [B, 576, D] — patch embeddings (CLS removed)
        The `remove_cls_token` flag is kept for backward compat but CLS is
        always returned separately so callers can use it for visual saliency.
        """
        if self.normalize:
            images = (images - self._mean) / self._std

        # LLaVA-1.5 uses CLIP vision features from the penultimate layer
        # (vision_feature_layer = -2). Using last_hidden_state here creates a
        # feature distribution mismatch with the pretrained projector.
        outputs = self.model(pixel_values=images, output_hidden_states=True)
        all_tokens = outputs.hidden_states[-2]  # [B, 1+N, D]

        cls_embedding  = all_tokens[:, 0, :]   # [B, D]
        patch_tokens   = all_tokens[:, 1:, :]  # [B, N, D]
        return cls_embedding, patch_tokens


class CLIPTextEncoder(nn.Module):
    """Wrapper around HuggingFace CLIP text encoder."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        transformers = _get_hf_transformers()

        self.model = transformers.CLIPTextModel.from_pretrained(model_name)
        self.config = self.model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


class LLaVALM(nn.Module):
    """
    Lightweight wrapper for a text-only causal LM that accepts projected visual
    tokens as a prefix through inputs_embeds.

    IMPORTANT:
    - This wrapper is for a custom multimodal pipeline:
        CLIP vision -> projector -> visual prefix -> text LM
    - `model_name` should be a text-only causal LM that works with
      AutoModelForCausalLM, not a full LLaVA checkpoint.
    """

    def __init__(
        self,
        model_name: str = "lmsys/vicuna-7b-v1.5",
        device: str = None,
        torch_dtype=None,
        use_grad: bool = False,
        scale_visual_prefix: bool = True,
        visual_prefix_scale: float = 1.0,
    ):
        super().__init__()
        import gc
        transformers = _get_hf_transformers()

        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype, None)

        # If model_name points to a full LLaVA checkpoint (e.g. llava-hf/llava-1.5-7b-hf),
        # extract only the language model. This avoids downloading Vicuna separately
        # when LLaVA-1.5 is already being used — saving ~13 GB of storage.
        if "llava" in model_name.lower():
            try:
                from transformers import LlavaForConditionalGeneration
            except ImportError:
                raise ImportError(
                    "transformers >= 4.37 required for LlavaForConditionalGeneration. "
                    "Run: pip install -U transformers"
                )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                legacy=False,
            )
            _llava_full = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="cpu",
            )
            self.model = _llava_full.language_model
            # Expose pretrained projector weights so PruningVLM can init from them.
            # We deepcopy to CPU to free the rest of the LLaVA checkpoint immediately.
            import copy
            self._llava_projector = copy.deepcopy(_llava_full.multi_modal_projector).cpu()
            del _llava_full
            gc.collect()
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                legacy=False,
            )
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=None,
            )


        self.hidden_size = self.model.config.hidden_size
        self.scale_visual_prefix = bool(scale_visual_prefix)
        self.visual_prefix_scale = float(visual_prefix_scale)
        self.use_grad = bool(use_grad)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ID of the <image> placeholder token used in LLaVA-1.5 prompts.
        # Visual tokens are inserted INLINE at this position (not prepended)
        # to match the format the LLM was pretrained with.
        _img_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.image_token_id = (
            _img_id
            if _img_id not in (None, self.tokenizer.unk_token_id)
            else 32000
        )

        # Freeze base model params if requested, but do NOT force eval() here.
        # This avoids conflicts when LoRA/adapters are attached later for training.
        if not self.use_grad:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(
        self,
        projected_visual_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        visual_attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        return_logits: bool = True,
        output_hidden_states: bool = False,
    ):
        input_embed_layer = self.model.get_input_embeddings()

        # Cast to LLM dtype. No magnitude rescaling: when using the pretrained
        # LLaVA-1.5 projector + frozen LLaVA-1.5 LLM, the projector already
        # outputs tokens at the correct scale. Rescaling distorts the distribution.
        projected_visual_tokens = projected_visual_tokens.to(
            device=input_ids.device,
            dtype=next(input_embed_layer.parameters()).dtype,
        )

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

        # ── Insert visual tokens INLINE at <image> token position ──────────
        # LLaVA-1.5 training format: "SYSTEM USER: <image>\n{question}\nASSISTANT:"
        # Per-sample processing handles variable padding correctly.
        img_tok_id = getattr(self, 'image_token_id', 32000)
        B, K_vis, D_vis = projected_visual_tokens.shape

        if visual_attention_mask is None:
            vis_mask_base = torch.ones(K_vis, dtype=attention_mask.dtype,
                                       device=attention_mask.device)
        else:
            vis_mask_base = visual_attention_mask[0].to(device=attention_mask.device,
                                                        dtype=attention_mask.dtype)

        embed_list, mask_list, lbl_list = [], [], []
        has_lbl = labels is not None
        if has_lbl:
            labels = labels.to(input_ids.device)

        for b in range(B):
            ids_b = input_ids[b]          # [L]
            positions = (ids_b == img_tok_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                p = positions[0].item()
                pre_e  = input_embed_layer(ids_b[:p].unsqueeze(0))     # [1, p, D]
                post_e = input_embed_layer(ids_b[p+1:].unsqueeze(0))   # [1, L-p-1, D]
                emb_b  = torch.cat([pre_e, projected_visual_tokens[b:b+1], post_e], dim=1)

                pre_m  = attention_mask[b, :p]
                post_m = attention_mask[b, p+1:]
                if visual_attention_mask is not None:
                    vm = visual_attention_mask[b].to(device=attention_mask.device,
                                                     dtype=attention_mask.dtype)
                else:
                    vm = vis_mask_base
                msk_b  = torch.cat([pre_m, vm, post_m], dim=0).unsqueeze(0)  # [1, L-1+K]

                if has_lbl:
                    pre_l  = labels[b, :p]
                    post_l = labels[b, p+1:]
                    vis_l  = torch.full((K_vis,), -100, dtype=labels.dtype, device=labels.device)
                    lbl_b  = torch.cat([pre_l, vis_l, post_l], dim=0).unsqueeze(0)
            else:
                # Fallback: prepend visual tokens before text
                text_e = input_embed_layer(ids_b.unsqueeze(0))  # [1, L, D]
                emb_b  = torch.cat([projected_visual_tokens[b:b+1], text_e], dim=1)
                vm     = vis_mask_base
                msk_b  = torch.cat([vm, attention_mask[b]], dim=0).unsqueeze(0)
                if has_lbl:
                    vis_l = torch.full((K_vis,), -100, dtype=labels.dtype, device=labels.device)
                    lbl_b = torch.cat([vis_l, labels[b]], dim=0).unsqueeze(0)

            embed_list.append(emb_b)
            mask_list.append(msk_b)
            if has_lbl:
                lbl_list.append(lbl_b)

        inputs_embeds       = torch.cat(embed_list, dim=0)   # [B, L-1+K, D]
        full_attention_mask = torch.cat(mask_list,  dim=0)   # [B, L-1+K]
        full_labels         = torch.cat(lbl_list,   dim=0) if has_lbl else None

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            return_dict=True,
            output_hidden_states=output_hidden_states,
        )

        return outputs

    def tokenize(self, texts, **kwargs):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            **kwargs,
        )