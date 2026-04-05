#!/usr/bin/env python3
"""
Detailed diagnostic of visual encoding and generation quality
"""
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPTokenizer

os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf_home/hub"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home/transformers"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "./test.jpg"
CHECKPOINT_PATH = "checkpoints/best_model.pt"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

from models.pruning_vlm import PruningVLM

def load_image_as_tensor(image_path: str, image_size: int = 224) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    image = torch.from_numpy(np.array(image)).float() / 255.0
    image = image.permute(2, 0, 1).contiguous()
    return image

def build_prompts(question: str):
    clip_instruction = question
    llm_prompt = f"USER: <image>\n{question}\nASSISTANT:"
    return clip_instruction, llm_prompt

print("=" * 100)
print("DETAILED VLM DIAGNOSTIC")
print("=" * 100)

# Load model
model = PruningVLM(
    clip_model_name=CLIP_MODEL_NAME,
    llm_model_name=LLM_MODEL_NAME,
    keep_ratio=0.5,
    alpha=0.6,
    learnable_alpha=False,
    llm_torch_dtype="bfloat16" if torch.cuda.is_available() else None,
).to(DEVICE)

ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt["model"]
model.load_state_dict(state_dict, strict=True)
model.eval()

# Load image
image = load_image_as_tensor(IMAGE_PATH, 224).unsqueeze(0).to(DEVICE)
print(f"\n1. IMAGE")
print(f"   Shape: {image.shape}")
print(f"   Min: {image.min():.4f}, Max: {image.max():.4f}, Mean: {image.mean():.4f}")
print(f"   Has NaN: {torch.isnan(image).any()}, Has Inf: {torch.isinf(image).any()}")

# Process through vision encoder
with torch.no_grad():
    cls_emb, visual_tokens = model.vision_encoder(image)
    print(f"\n2. VISION ENCODER OUTPUT")
    print(f"   Shape: {visual_tokens.shape}")
    print(f"   Min: {visual_tokens.min():.6f}, Max: {visual_tokens.max():.6f}, Mean: {visual_tokens.mean():.6f}")
    print(f"   Std: {visual_tokens.std():.6f}")
    print(f"   Has NaN: {torch.isnan(visual_tokens).any()}, Has Inf: {torch.isinf(visual_tokens).any()}")
    
    # Project to LLM space
    z = model.projector(visual_tokens)
    print(f"\n3. PROJECTOR OUTPUT (z)")
    print(f"   Shape: {z.shape}")
    print(f"   Min: {z.min():.6f}, Max: {z.max():.6f}, Mean: {z.mean():.6f}")
    print(f"   Std: {z.std():.6f}")
    print(f"   Has NaN: {torch.isnan(z).any()}, Has Inf: {torch.isinf(z).any()}")
    
    # Get text tokens
    clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
    question = "What is in the image?"
    clip_instruction, llm_prompt = build_prompts(question)
    
    clip_tokens = clip_tokenizer(
        [clip_instruction],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    clip_input_ids = clip_tokens.input_ids.to(DEVICE)
    clip_attention_mask = clip_tokens.attention_mask.to(DEVICE)
    
    llm_tokens = model.llm.tokenizer(
        [llm_prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    llm_input_ids = llm_tokens.input_ids.to(DEVICE)
    llm_attention_mask = llm_tokens.attention_mask.to(DEVICE)
    
    print(f"\n4. TEXT TOKENS")
    print(f"   LLM Prompt: '{llm_prompt}'")
    print(f"   LLM Input IDs shape: {llm_input_ids.shape}")
    print(f"   LLM Tokens (first 15): {llm_input_ids[0, :15].tolist()}")
    
    # Text encoder
    text_tokens = model.text_encoder(clip_input_ids, clip_attention_mask)
    print(f"\n5. TEXT ENCODER OUTPUT")
    print(f"   Shape: {text_tokens.shape}")
    print(f"   Min: {text_tokens.min():.6f}, Max: {text_tokens.max():.6f}, Mean: {text_tokens.mean():.6f}")
    print(f"   Std: {text_tokens.std():.6f}")
    print(f"   Has NaN: {torch.isnan(text_tokens).any()}, Has Inf: {torch.isinf(text_tokens).any()}")
    
    # Get LLM embeddings
    text_embeds = model.llm.model.get_input_embeddings()(llm_input_ids)
    print(f"\n6. LLM TEXT EMBEDDINGS")
    print(f"   Shape: {text_embeds.shape}")
    print(f"   Min: {text_embeds.min():.6f}, Max: {text_embeds.max():.6f}, Mean: {text_embeds.mean():.6f}")
    print(f"   Std: {text_embeds.std():.6f}")
    print(f"   Has NaN: {torch.isnan(text_embeds).any()}, Has Inf: {torch.isinf(text_embeds).any()}")
    
    # Test scale matching and damping (as done in infer_debug.py)
    visual_for_llm = z.to(device=text_embeds.device, dtype=text_embeds.dtype)
    vis_scale = visual_for_llm.abs().mean(dim=(1, 2), keepdim=True) + 1e-8
    txt_scale = text_embeds.abs().mean(dim=(1, 2), keepdim=True) + 1e-8
    
    print(f"\n7. SCALE ANALYSIS")
    print(f"   Visual scale (before adjustment): {vis_scale.squeeze():.6f}")
    print(f"   Text scale: {txt_scale.squeeze():.6f}")
    print(f"   Scale ratio (txt/vis): {(txt_scale / vis_scale).squeeze():.6f}")
    
    visual_for_llm = visual_for_llm * (txt_scale / vis_scale)
    print(f"   Visual after scale matching: abs_mean={visual_for_llm.abs().mean():.6f}")
    
    visual_for_llm = 0.2 * visual_for_llm
    print(f"   Visual after 0.2x damping: abs_mean={visual_for_llm.abs().mean():.6f}")
    
    # Check if visual embeddings are essentially zero
    ratio = visual_for_llm.abs().mean() / text_embeds.abs().mean()
    print(f"   Visual/Text ratio: {ratio:.6f}")
    if ratio < 0.1:
        print(f"   ⚠ WARNING: Visual embeddings are much smaller than text (ratio < 0.1)")
    
    print(f"\n8. COMBINED EMBEDDINGS")
    inputs_embeds = torch.cat([visual_for_llm, text_embeds], dim=1)
    print(f"   Shape: {inputs_embeds.shape}")
    print(f"   Min: {inputs_embeds.min():.6f}, Max: {inputs_embeds.max():.6f}, Mean: {inputs_embeds.mean():.6f}")
    print(f"   Std: {inputs_embeds.std():.6f}")
    print(f"   First 5 token means (visual): {visual_for_llm[0, :5, :].mean(dim=1).tolist()}")
    print(f"   First 5 token means (text): {text_embeds[0, :5, :].mean(dim=1).tolist()}")

print("\n" + "=" * 100)
print("END DIAGNOSTIC")
print("=" * 100)
