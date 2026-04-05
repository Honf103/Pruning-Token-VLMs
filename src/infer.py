import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from models.pruning_vlm import PruningVLM


# ------------------------------------------------------------
# Stable HF cache path
# ------------------------------------------------------------
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf_home/hub"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home/transformers"


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_PATH = "test.jpg"
CHECKPOINT_PATH = "checkpoints/best_model/training_state.pt"

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
LLM_MODEL_NAME = "lmsys/vicuna-7b-v1.5"

# Must match training setup as much as possible
ALPHA = 0.6
KEEP_RATIO = 0.7
IMAGE_SIZE = 336
MAX_NEW_TOKENS = 64

# LoRA config must match train
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# ------------------------------------------------------------
# Image preprocess
# NOTE:
# Ideally this should match your training image preprocessing exactly.
# ------------------------------------------------------------
def load_image_as_tensor(image_path: str, image_size: int = 224) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous()  # [3, H, W]
    return image


# ------------------------------------------------------------
# Prompt
# IMPORTANT:
# Prompt format should match llm_texts during training
# ------------------------------------------------------------
def build_prompt(question: str):
    clip_instruction = question                   # for CLIP text encoder
    _VICUNA_SYSTEM = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    llm_prompt = f"{_VICUNA_SYSTEM} USER: {question} ASSISTANT:"  # must match train.py prompt format
    return clip_instruction, llm_prompt


# ------------------------------------------------------------
# LoRA attach
# ------------------------------------------------------------
def attach_lora_to_llm(
    llm_wrapper,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    if not hasattr(llm_wrapper, "model"):
        raise ValueError("Expected model.llm to have `.model` attribute.")

    base_llm = llm_wrapper.model

    for p in base_llm.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    peft_llm = get_peft_model(base_llm, lora_config)
    llm_wrapper.model = peft_llm
    return llm_wrapper


def safe_get_alpha(model: PruningVLM) -> float:
    if hasattr(model, "score_fusion") and hasattr(model.score_fusion, "alpha"):
        return torch.clamp(model.score_fusion.alpha.detach(), 0.0, 1.0).item()
    return float(ALPHA)


# ------------------------------------------------------------
# Main inference
# ------------------------------------------------------------
@torch.no_grad()
def main():
    if not Path(IMAGE_PATH).exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    model = PruningVLM(
        clip_model_name=CLIP_MODEL_NAME,
        llm_model_name=LLM_MODEL_NAME,
        keep_ratio=KEEP_RATIO,
        alpha=ALPHA,
        learnable_alpha=True,
        llm_torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        gate_temperature=0.5,
        use_ste=False,
        gate_threshold_mode="topk",
        projector_hidden_dim=2048,
        ia_hidden_dim=512,
    )

    # --------------------------------------------------------
    # Attach LoRA before loading checkpoint
    # --------------------------------------------------------
    model.llm = attach_lora_to_llm(
        model.llm,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Load checkpoint — same format as save_lora_and_non_llm_trainables()
    # non_llm_trainables.pt  → projector, instruction_aware, etc.
    # llm_lora_adapter/      → LoRA weights
    # --------------------------------------------------------
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    non_llm_path = os.path.join(checkpoint_dir, "non_llm_trainables.pt")
    lora_dir = os.path.join(checkpoint_dir, "llm_lora_adapter")

    if os.path.exists(non_llm_path):
        non_llm_ckpt = torch.load(non_llm_path, map_location=DEVICE)
        ckpt_state = non_llm_ckpt["trainable_non_llm_state_dict"]
        model_state = model.state_dict()
        compatible = {k: v for k, v in ckpt_state.items()
                      if k in model_state and model_state[k].shape == v.shape}
        skipped = [k for k in ckpt_state if k not in compatible]
        model.load_state_dict(compatible, strict=False)
        print(f"Loaded non-LLM weights: {len(compatible)} keys. Skipped: {len(skipped)}")
    else:
        print(f"[WARN] non_llm_trainables.pt not found at: {non_llm_path}")

    if os.path.exists(lora_dir):
        model.llm.model.load_adapter(lora_dir, adapter_name="default")
        print(f"Loaded LoRA adapter from: {lora_dir}")
    else:
        print(f"[WARN] LoRA adapter not found at: {lora_dir}")

    print(f"Loaded checkpoint from: {checkpoint_dir}")

    model.eval()

    # --------------------------------------------------------
    # Tokenizers
    # --------------------------------------------------------
    clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
    llm_tokenizer = model.llm.tokenizer

    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------
    question = "how many car in the image?"
    clip_instruction, llm_prompt = build_prompt(question)

    image = load_image_as_tensor(IMAGE_PATH, IMAGE_SIZE).unsqueeze(0).to(DEVICE)

    clip_tokens = clip_tokenizer(
        [clip_instruction],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    clip_input_ids = clip_tokens.input_ids.to(DEVICE)
    clip_attention_mask = clip_tokens.attention_mask.to(DEVICE)

    llm_tokens = llm_tokenizer(
        [llm_prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    llm_input_ids = llm_tokens.input_ids.to(DEVICE)
    llm_attention_mask = llm_tokens.attention_mask.to(DEVICE)

    # --------------------------------------------------------
    # Forward through pruning model in hard-pruning mode
    # --------------------------------------------------------
    outputs = model(
        images=image,
        input_ids=clip_input_ids,
        attention_mask=clip_attention_mask,
        llm_input_ids=llm_input_ids,
        llm_attention_mask=llm_attention_mask,
        labels=None,
        return_intermediates=True,
        use_hard_pruning=True,
    )

    z = outputs["projected_tokens"]              # [1, N, D]
    visual_for_llm = outputs["visual_for_llm"]   # [1, K, D]
    kept_indices = outputs["kept_indices"]       # [1, K]
    kept_scores = outputs["kept_scores"]         # [1, K]
    fused_scores = outputs["fused_scores"]       # [1, N]

    # --------------------------------------------------------
    # Build inputs_embeds for LLM
    # --------------------------------------------------------
    text_embeds = model.llm.model.get_input_embeddings()(llm_input_ids)

    visual_for_llm = visual_for_llm.to(
        device=text_embeds.device,
        dtype=text_embeds.dtype,
    )

    # Optional scale matching
    vis_scale = visual_for_llm.abs().mean(dim=(1, 2), keepdim=True) + 1e-8
    txt_scale = text_embeds.abs().mean(dim=(1, 2), keepdim=True) + 1e-8
    visual_for_llm = visual_for_llm * (txt_scale / vis_scale)

    inputs_embeds = torch.cat([visual_for_llm, text_embeds], dim=1)

    visual_mask = torch.ones(
        visual_for_llm.shape[:2],
        dtype=llm_attention_mask.dtype,
        device=llm_attention_mask.device,
    )
    full_attention_mask = torch.cat([visual_mask, llm_attention_mask], dim=1)

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------
    generated_ids = model.llm.model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=full_attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        pad_token_id=llm_tokenizer.pad_token_id,
        eos_token_id=llm_tokenizer.eos_token_id,
        use_cache=True,
    )

    answer = llm_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    # --------------------------------------------------------
    # Print debug info
    # --------------------------------------------------------
    score_mean = fused_scores.mean().item()
    score_std = fused_scores.std().item()
    score_min = fused_scores.min().item()
    score_max = fused_scores.max().item()

    print("=" * 80)
    print("Question:", question)
    print("Learned alpha:", round(safe_get_alpha(model), 4))
    print("Keep ratio:", KEEP_RATIO)
    print("Original projected tokens:", z.size(1))
    print("Kept tokens:", visual_for_llm.size(1))
    print("Fused score mean/std:", round(score_mean, 4), "/", round(score_std, 4))
    print("Fused score min/max:", round(score_min, 4), "/", round(score_max, 4))
    print("Answer:")
    print(answer)
    print("=" * 80)

    print("Top kept indices:", kept_indices[0].tolist()[:20])
    print("Top kept scores:", [round(x, 4) for x in kept_scores[0].tolist()[:20]])


if __name__ == "__main__":
    main()