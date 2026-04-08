"""
visualize_pruning.py
--------------------
Run inference with "Who is the most handsome?" on test_1.jpg at
keep_ratio = 0.2, 0.6, 0.8 (token merging ON) and display:

  - Original image
  - Per-ratio: heatmap overlay of token importance scores
                + binary keep/drop outline
  - Response text for each ratio

Layout (matplotlib):
  Row 0 : original image
  Row 1-3: one row per ratio — heatmap overlay | response

Usage:
  cd /workspace/VLM_Project/src
  python visualize_pruning.py
  python visualize_pruning.py --image test_1.jpg --ratios 0.2 0.6 0.8

This visualizer always runs with token merging enabled.
"""

import os, sys, argparse
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # headless / file output
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from PIL import Image
from transformers import CLIPTokenizer
from peft import LoraConfig, TaskType, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval import load_lightweight_checkpoint, prepare_images_for_model, generate_answers, build_image_transform
from utils.misc import set_seed

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints", "best_model_fastbest")
IMG_PATH   = os.path.join(BASE_DIR, "test_1.jpg")
PROMPT     = "How many man are in the image?"
PATCH_SIZE = 14          # CLIP ViT-L/14
GRID_SIZE  = 24          # 336 / 14 = 24
N_TOKENS   = GRID_SIZE * GRID_SIZE   # 576

VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


# ─────────────────────────────────────────────────────────────
# Build / load model
# ─────────────────────────────────────────────────────────────
def build_model(keep_ratio, use_merging, device, use_bf16):
    from models.pruning_vlm import PruningVLM

    model = PruningVLM(
        clip_model_name="openai/clip-vit-large-patch14-336",
        llm_model_name="llava-hf/llava-1.5-7b-hf",
        keep_ratio=keep_ratio,
        alpha=0.6,
        learnable_alpha=True,
        remove_cls_token=True,
        llm_torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        gate_temperature=0.2,
        gate_score_scale=5.0,
        gate_threshold_mode="topk",
        llm_use_grad=False,
        projector_hidden_dim=2048,
        ia_hidden_dim=512,
        use_merging=use_merging,
    )
    for p in model.vision_encoder.parameters():
        p.requires_grad = False
    for p in model.text_encoder.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    )
    model.llm.model = get_peft_model(model.llm.model, lora_config)
    model, _ = load_lightweight_checkpoint(model, CKPT_DIR, device)
    model = model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, image_pil, keep_ratio, device, use_bf16,
                  clip_tokenizer, llm_tokenizer,
                  max_new_tokens=64, num_beams=1):
    model.set_keep_ratio(keep_ratio)

    images_t = prepare_images_for_model([image_pil], device=device)

    llm_prompt = f"{VICUNA_SYSTEM} USER: <image>\n{PROMPT}\nASSISTANT:"

    clip_enc = clip_tokenizer(
        [PROMPT], return_tensors="pt",
        padding=True, truncation=True, max_length=64,
    )
    llm_enc = llm_tokenizer(
        [llm_prompt], return_tensors="pt",
        padding=True, truncation=True, max_length=256,
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        enabled=torch.cuda.is_available(),
    ):
        preds, info = generate_answers(
            model=model,
            images=images_t,
            clip_input_ids=clip_enc.input_ids.to(device),
            clip_attention_mask=clip_enc.attention_mask.to(device),
            prompt_input_ids=llm_enc.input_ids.to(device),
            prompt_attention_mask=llm_enc.attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            use_hard_pruning=True,
        )

    response   = preds[0]
    scores_1d  = info["fused_scores"][0].float().cpu().numpy()   # [N]
    hard_mask  = info["hard_mask"][0].float().cpu().numpy()       # [N]
    kept_count = int(hard_mask.sum())

    return response, scores_1d, hard_mask, kept_count


# ─────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────
def scores_to_heatmap(scores_1d, grid=GRID_SIZE):
    """Reshape flat scores → (grid, grid) float32 image, normalised 0-1."""
    n = len(scores_1d)
    if n != grid * grid:
        # Pad / truncate to fit grid
        padded = np.zeros(grid * grid, dtype=np.float32)
        padded[:min(n, grid*grid)] = scores_1d[:min(n, grid*grid)]
        scores_1d = padded
    grid_scores = scores_1d.reshape(grid, grid).astype(np.float32)
    mn, mx = grid_scores.min(), grid_scores.max()
    if mx > mn:
        grid_scores = (grid_scores - mn) / (mx - mn)
    return grid_scores


def overlay_heatmap(ax, image_pil, scores_1d, hard_mask, keep_ratio, grid=GRID_SIZE):
    """
    Display:
      - preprocessed 336×336 image (exact model input)
      - coloured heatmap overlay (jet, alpha=0.45) of fused_scores
      - dark overlay on DROPPED patches
    """
    img_np = np.array(image_pil)   # already 336×336 after preprocess_image()

    ax.imshow(img_np)

    # Heatmap overlay
    heat = scores_to_heatmap(scores_1d, grid)
    ax.imshow(
        heat,
        cmap="jet",
        alpha=0.40,
        extent=[0, 336, 336, 0],   # match image coordinates
        interpolation="nearest",
        vmin=0, vmax=1,
    )

    # Thin outlines: white = kept, dark = dropped
    drop_mask = (hard_mask == 0)
    ps = PATCH_SIZE
    for idx in np.where(drop_mask)[0]:
        row, col = divmod(int(idx), grid)
        rect = patches.Rectangle(
            (col * ps, row * ps), ps, ps,
            linewidth=0.4, edgecolor="#222222", facecolor="black", alpha=0.45,
        )
        ax.add_patch(rect)

    kept_count = int(hard_mask.sum())
    ax.set_title(
        f"kr={keep_ratio:.1f}  ({kept_count}/{grid*grid} tokens kept)",
        fontsize=9, pad=3,
    )
    ax.axis("off")


def save_keep_map_figure(image_pil, results, use_merging, output_path,
                         orig_size=None, grid=GRID_SIZE):
    """
    Slide-friendly figure: horizontal strip, one panel per keep-ratio.
    image_pil is already 336×336 (preprocessed to match training).
    Kept patches → original image visible.
    Dropped patches → black overlay (alpha 0.78) so they fade out clearly.
    """
    n = len(results)
    fig, axes = plt.subplots(
        1, n + 1,
        figsize=(3.6 * (n + 1), 4.2),
        gridspec_kw={"wspace": 0.06},
    )

    img_np = np.array(image_pil)   # already 336×336 after preprocess_image()

    # ── Column 0: model input (preprocessed) ────────────────
    orig_label = f"({orig_size[0]}×{orig_size[1]} → 336×336)" if orig_size else "(336×336)"
    axes[0].imshow(img_np)
    axes[0].set_title(f"Model Input\n{orig_label}", fontsize=11, fontweight="bold", pad=6)
    axes[0].axis("off")

    ps = PATCH_SIZE  # 14 px per patch in 336-px image

    for col, r in enumerate(results, start=1):
        ax = axes[col]
        ax.imshow(img_np)

        # Build a single RGBA overlay: kept = transparent, dropped = dark
        overlay = np.zeros((336, 336, 4), dtype=np.float32)
        drop_mask = (r["mask"] == 0)
        for idx in np.where(drop_mask)[0]:
            row_i, col_i = divmod(int(idx), grid)
            y0, x0 = row_i * ps, col_i * ps
            overlay[y0:y0+ps, x0:x0+ps, :] = [0.0, 0.0, 0.0, 0.78]

        ax.imshow(overlay, interpolation="nearest")

        kept   = r["kept"]
        total  = grid * grid
        pct    = kept / total * 100
        merge_tag = "(merge)" if use_merging else ""
        ax.set_title(
            f"keep ratio = {r['kr']:.1f}  {merge_tag}\n"
            f"{kept}/{total} tokens  ({pct:.0f}%)",
            fontsize=10, fontweight="bold", pad=6,
            color="#1a1a7e",
        )

        # Response label at bottom
        resp_short = r["response"][:60] + ("…" if len(r["response"]) > 60 else "")
        ax.text(
            0.5, -0.04,
            f'→ "{resp_short}"',
            fontsize=8.5, color="#333333",
            ha="center", va="top",
            transform=ax.transAxes,
            style="italic",
        )
        ax.axis("off")

    fig.suptitle(
        f'Token Keep Map  —  Prompt: "{PROMPT}"',
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Shared legend patch
    import matplotlib.patches as mpatches
    kept_patch    = mpatches.Patch(facecolor="#ffffff", edgecolor="#555", label="Kept token")
    dropped_patch = mpatches.Patch(facecolor="#222222", edgecolor="#555", label="Dropped token")
    fig.legend(
        handles=[kept_patch, dropped_patch],
        loc="lower center", ncol=2,
        fontsize=9, framealpha=0.85,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"  Saved keep-map → {output_path}")
    plt.close(fig)


def wrap_text(text, width=55):
    """Simple word wrapper."""
    words = text.split()
    lines, line = [], []
    for w in words:
        if sum(len(x)+1 for x in line) + len(w) > width:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",   default=IMG_PATH)
    p.add_argument("--ratios",  type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2, 0.4, 1.0])
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--output",  default=os.path.join(BASE_DIR, "pruning_vis.png"))
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def preprocess_image(path: str, image_size: int = 336) -> tuple:
    """
    Load & preprocess image to match the training pipeline:
      Resize(image_size × image_size) → ToTensor() [0,1] → back to PIL.
    Returns (original_pil, preprocessed_pil).
    CLIP mean/std normalisation is applied inside CLIPVisionEncoder.forward().
    """
    from torchvision import transforms as T
    image_raw = Image.open(path).convert("RGB")
    tfm = build_image_transform(image_size=image_size)
    img_tensor = tfm(image_raw)                          # [3, H, W] float [0,1]
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image_preprocessed = Image.fromarray(img_np)        # 336×336 PIL, exact model input
    return image_raw, image_preprocessed


def main():
    args = parse_args()
    set_seed(args.seed)

    use_merging = True
    ratios      = sorted(args.ratios)
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16    = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    image_pil_raw, image_pil = preprocess_image(args.image, image_size=336)

    print(f"\n  Image  : {args.image}")
    print(f"  Original size : {image_pil_raw.size[0]}×{image_pil_raw.size[1]}")
    print(f"  Preprocessed  : {image_pil.size[0]}×{image_pil.size[1]}  (resize → ToTensor, matches training)")
    print(f"  Checkpoint: {CKPT_DIR}")
    print(f"  Prompt : {PROMPT}")
    print(f"  Ratios : {ratios}")
    print("  Merging: True (forced in visualize_pruning.py)")
    print(f"\n  Loading model …")

    model = build_model(
        keep_ratio=ratios[0],
        use_merging=use_merging,
        device=device,
        use_bf16=use_bf16,
    )

    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")
    llm_tokenizer  = model.llm.tokenizer
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    # ── Collect results ──────────────────────────────────────
    results = []
    for kr in ratios:
        print(f"\n  Inference  kr={kr:.1f} …", flush=True)
        response, scores_1d, hard_mask, kept = run_inference(
            model=model,
            image_pil=image_pil,
            keep_ratio=kr,
            device=device,
            use_bf16=use_bf16,
            clip_tokenizer=clip_tokenizer,
            llm_tokenizer=llm_tokenizer,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"    kept={kept}  response: {response[:80]}")
        results.append({"kr": kr, "response": response,
                        "scores": scores_1d, "mask": hard_mask, "kept": kept})

    # ── Plot ─────────────────────────────────────────────────
    n_ratios = len(ratios)
    fig_h    = 2.0 + n_ratios * 3.6
    fig, axes = plt.subplots(
        n_ratios + 1, 2,
        figsize=(10, fig_h),
        gridspec_kw={"width_ratios": [1, 1.6], "hspace": 0.45, "wspace": 0.12},
    )

    # Row 0: original vs preprocessed image
    ax_orig_img  = axes[0, 0]
    ax_orig_text = axes[0, 1]

    orig_w, orig_h = image_pil_raw.size
    ax_orig_img.imshow(np.array(image_pil))   # already 336×336 after preprocess
    ax_orig_img.set_title(
        f"Model input (336×336)\n[original: {orig_w}×{orig_h} → resize → ToTensor]",
        fontsize=8.5, pad=3,
    )
    ax_orig_img.axis("off")

    ax_orig_text.axis("off")
    ax_orig_text.text(
        0.02, 0.60,
        f'Prompt:\n"{PROMPT}"',
        fontsize=11, va="top", ha="left",
        transform=ax_orig_text.transAxes,
        fontstyle="italic", color="#333333",
    )
    ax_orig_text.text(
        0.02, 0.30,
        f"Model: LLaVA-1.5-7B + IA-pruner\nMerging: {'ON' if use_merging else 'OFF'}\n"
        f"Preprocess: Resize(336×336) → ToTensor → CLIP norm (inside encoder)",
        fontsize=8, va="top", ha="left",
        transform=ax_orig_text.transAxes, color="#666666",
    )

    # Rows 1…n: one per ratio
    for i, r in enumerate(results):
        ax_img  = axes[i + 1, 0]
        ax_text = axes[i + 1, 1]

        overlay_heatmap(ax_img, image_pil, r["scores"], r["mask"], r["kr"])

        # Response panel
        ax_text.axis("off")
        merge_tag = "merge ON" if use_merging else "merge OFF"
        header    = f"kr={r['kr']:.1f}  ({r['kept']}/{N_TOKENS} tokens)  [{merge_tag}]"
        body      = wrap_text(r["response"], width=58)

        ax_text.text(
            0.02, 0.95, header,
            fontsize=8.5, va="top", ha="left",
            transform=ax_text.transAxes,
            fontweight="bold", color="#1a1a7e",
        )
        ax_text.text(
            0.02, 0.78, body,
            fontsize=9.5, va="top", ha="left",
            transform=ax_text.transAxes,
            color="#111111", linespacing=1.5,
            wrap=True,
        )

    # Colour-bar legend (shared)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:, 0], fraction=0.025, pad=0.02, aspect=35)
    cbar.set_label("Token importance (normalised)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        f'Token pruning visualisation  —  "{PROMPT}"',
        fontsize=11, y=1.005, fontweight="bold",
    )

    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\n  Saved → {args.output}")

    # ── Slide-friendly keep-map figure ───────────────────────
    keepmap_path = args.output.replace(".png", "_keepmap.png")
    save_keep_map_figure(image_pil, results, use_merging, keepmap_path,
                         orig_size=image_pil_raw.size)

    # Also print clean summary
    print("\n" + "=" * 60)
    print(f"  Prompt: {PROMPT}")
    print("=" * 60)
    for r in results:
        merge_tag = "(merge)" if use_merging else ""
        print(f"\n  kr={r['kr']:.1f} {merge_tag}  [{r['kept']}/{N_TOKENS} tokens]")
        print(f"  Response: {r['response']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
