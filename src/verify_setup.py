#!/usr/bin/env python3
"""
Comprehensive diagnosis of the VLM inference setup
"""
import os
import sys
from pathlib import Path

print("=" * 100)
print("SETUP VERIFICATION")
print("=" * 100)

# Check environment
print("\n1. DEVICE & ENVIRONMENT")
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {DEVICE}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Check imports
print("\n2. IMPORTS")
try:
    from transformers import CLIPTokenizer
    print("   ✓ transformers (CLIPTokenizer)")
except Exception as e:
    print(f"   ✗ transformers: {e}")

try:
    from PIL import Image
    print("   ✓ PIL (Image)")
except Exception as e:
    print(f"   ✗ PIL: {e}")

try:
    from models.pruning_vlm import PruningVLM
    print("   ✓ PruningVLM")
except Exception as e:
    print(f"   ✗ PruningVLM: {e}")

# Check files
print("\n3. FILES")
files_to_check = [
    ("test.jpg", "./test.jpg"),
    ("best_model.pt", "checkpoints/best_model.pt"),
    ("infer_debug.py", "./infer_debug.py"),
]

for name, path in files_to_check:
    exists = Path(path).exists()
    size = f"{Path(path).stat().st_size / 1024 / 1024:.2f} MB" if exists else "N/A"
    status = "✓" if exists else "✗"
    print(f"   {status} {name:20s} ({path:30s}) - {size}")

# Check model files
print("\n4. MODEL FILES")
model_files = [
    "cls_scorer.py",
    "hf_backbones.py",
    "instruction_aware.py",
    "projector.py",
    "pruning_vlm.py",
    "score_fusion.py",
    "text_importance.py",
    "token_pruner.py",
]

for f in model_files:
    exists = Path(f"models/{f}").exists()
    status = "✓" if exists else "✗"
    print(f"   {status} {f}")

# Test image
print("\n5. TEST IMAGE")
try:
    from PIL import Image
    img = Image.open("test.jpg")
    print(f"   ✓ Image loaded: {img.size}, mode={img.mode}")
except Exception as e:
    print(f"   ✗ Failed to load image: {e}")

# Test model loading
print("\n6. MODEL INITIALIZATION")
try:
    from models.pruning_vlm import PruningVLM
    
    model = PruningVLM(
        clip_model_name="openai/clip-vit-base-patch32",
        llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        keep_ratio=0.5,
        alpha=0.6,
        learnable_alpha=False,
        llm_torch_dtype="bfloat16" if torch.cuda.is_available() else None,
    ).to(DEVICE)
    print("   ✓ Model created successfully")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test checkpoint loading
print("\n7. CHECKPOINT LOADING")
try:
    ckpt = torch.load("checkpoints/best_model.pt", map_location=DEVICE)
    print(f"   ✓ Checkpoint loaded")
    print(f"   Keys: {list(ckpt.keys())}")
    
    state_dict_name = "model_state_dict" if "model_state_dict" in ckpt else "model"
    state_dict = ckpt[state_dict_name]
    print(f"   Using key: {state_dict_name}")
    print(f"   State dict keys: {len(state_dict.keys())} parameters")
    
    # Try loading into model
    model.load_state_dict(state_dict, strict=True)
    print("   ✓ Model state loaded successfully (strict=True)")
    model.eval()
except Exception as e:
    print(f"   ✗ Checkpoint loading failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 100)
print("VERIFICATION COMPLETE")
print("=" * 100)
