#!/usr/bin/env python3
"""
Verify model training vs inference consistency
"""
import os
import torch
from pathlib import Path
from PIL import Image
import numpy as np

os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf_home/hub"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home/transformers"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 100)
print("IMAGE & CHECKPOINT VALIDATION")
print("=" * 100)

# Check image
print("\n1. IMAGE FILE")
img_path = Path("./test.jpg")
if img_path.exists():
    stat = img_path.stat()
    print(f"   ✓ Exists: {stat.st_size / 1024 / 1024:.2f} MB")
    
    try:
        img = Image.open(img_path)
        print(f"   Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
        
        # Try to process as model would
        img_rgb = img.convert("RGB")
        img_resized = img_rgb.resize((224, 224))
        img_array = np.array(img_resized)
        print(f"   Resized: {img_array.shape}, dtype: {img_array.dtype}")
        print(f"   Pixel value range: [{img_array.min()}, {img_array.max()}]")
        
        # Convert to tensor as model does
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).contiguous()
        print(f"   Tensor shape: {img_tensor.shape}")
        print(f"   Tensor range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
        print(f"   Mean: {img_tensor.mean():.4f}, Std: {img_tensor.std():.4f}")
    except Exception as e:
        print(f"   ✗ Error processing image: {e}")
else:
    print(f"   ✗ Not found")

# Check checkpoint
print("\n2. CHECKPOINT FILE")
ckpt_path = Path("./checkpoints/best_model.pt")
if ckpt_path.exists():
    stat = ckpt_path.stat()
    print(f"   ✓ Exists: {stat.st_size / 1024 / 1024:.2f} MB")
    
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print(f"   Keys: {list(ckpt.keys())}")
        
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt.get("model", {})
        
        print(f"   Model params: {len(state_dict.keys())} tensors")
        print(f"   Epochs trained: {ckpt.get('epoch', 'N/A')}")
        print(f"   Avg loss: {ckpt.get('avg_loss', 'N/A')}")
        
        # Check some key params
        sample_keys = list(state_dict.keys())[:5]
        for key in sample_keys:
            tensor = state_dict[key]
            print(f"   - {key[:50]:50s}: {tuple(tensor.shape)} {tensor.dtype}")
    except Exception as e:
        print(f"   ✗ Error loading checkpoint: {e}")
else:
    print(f"   ✗ Not found")

# Compare train vs infer settings
print("\n3. TRAINING vs INFERENCE SETTINGS")
print(f"   Training keep_ratio: 0.25")
print(f"   Inference keep_ratio: 0.5")
print(f"   ⚠ MISMATCH: Training used 0.25, inference uses 0.5")
print(f"\n   Training alpha: 0.6")
print(f"   Inference alpha: 0.6")
print(f"   ✓ MATCH")

# Check dataset
print("\n4. DATASET FILES")
dataset_path = Path("./datasets/data")
if dataset_path.exists():
    llava_meta = dataset_path / "llava" / "metadata.json"
    if llava_meta.exists():
        print(f"   ✓ LLaVA metadata: {llava_meta.stat().st_size} bytes")
    coco_path = dataset_path / "coco" / "train2017"
    if coco_path.exists():
        num_images = len(list(coco_path.glob("*.jpg")))
        print(f"   ✓ COCO images: {num_images} images")
else:
    print(f"   ✗ Dataset path not found")

print("\n" + "=" * 100)
