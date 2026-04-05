"""
diagnose_vram.py -- VRAM anomaly diagnostic for PruningVLM

Investigates why kr=0.4 has higher peak VRAM than kr=0.7 in sweep results.

Runs inference with FIXED batch sizes at each kr to measure the actual
MB-per-sample cost (vs the calibration table in test_pope/test_mme).

Usage:
  python diagnose_vram.py
  python diagnose_vram.py --kr_list 1.0 0.7 0.4 0.1 --batch_sizes 4 16
"""

import os
import sys
import argparse
import math
import gc
import json

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval import load_lightweight_checkpoint, prepare_images_for_model, generate_answers
from test_pope import build_and_load_model
from utils.misc import set_seed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "best_model")

# Current (possibly wrong) calibration table in test_pope/test_mme
_CALIBRATED_MB_PER_SAMPLE = {
    1.0: 800,
    0.7: 600,
    0.4: 390,
    0.2: 255,
    0.1: 190,
}


def mb(n_bytes):
    return n_bytes / 1024 ** 2


def get_dummy_batch(batch_size, device, clip_tokenizer, llm_tokenizer,
                    clip_max_len=64, llm_max_len=256):
    """Synthetic random images -- token counts are deterministic regardless of content."""
    import numpy as np
    rng = np.random.default_rng(42)
    images = [
        Image.fromarray(rng.integers(0, 256, (336, 336, 3), dtype=np.uint8)).convert("RGB")
        for _ in range(batch_size)
    ]

    question = "Is there a dog in the image?"
    VICUNA_SYS = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    prompt = f"{VICUNA_SYS} USER: {question} Answer with yes or no. ASSISTANT:"

    images_t = prepare_images_for_model(images, device=device)

    clip_enc = clip_tokenizer(
        [question] * batch_size, return_tensors="pt",
        padding=True, truncation=True, max_length=clip_max_len,
    )
    llm_enc = llm_tokenizer(
        [prompt] * batch_size, return_tensors="pt",
        padding=True, truncation=True, max_length=llm_max_len,
    )

    return (
        images_t,
        clip_enc.input_ids.to(device),
        clip_enc.attention_mask.to(device),
        llm_enc.input_ids.to(device),
        llm_enc.attention_mask.to(device),
    )


@torch.no_grad()
def measure_vram(model, batch_size, device, clip_tokenizer, llm_tokenizer):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

    baseline_mb = mb(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0.0

    imgs, cids, cmask, pids, pmask = get_dummy_batch(
        batch_size, device, clip_tokenizer, llm_tokenizer
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        enabled=torch.cuda.is_available(),
    ):
        preds, info = generate_answers(
            model=model,
            images=imgs,
            clip_input_ids=cids,
            clip_attention_mask=cmask,
            prompt_input_ids=pids,
            prompt_attention_mask=pmask,
            max_new_tokens=8,
            num_beams=1,
            use_hard_pruning=True,
        )

    if torch.cuda.is_available():
        peak_mb_abs = mb(torch.cuda.max_memory_allocated())
        overhead_mb = peak_mb_abs - baseline_mb
        mb_per_sample = overhead_mb / batch_size
        torch.cuda.empty_cache()
    else:
        peak_mb_abs = overhead_mb = mb_per_sample = 0.0

    hard_mask = info["hard_mask"]          # [B, N]
    z         = info["projected_tokens"]   # [B, N, D]
    kept      = hard_mask.sum(dim=1).float().mean().item()
    total     = z.shape[1]

    return {
        "baseline_mb":  round(baseline_mb, 1),
        "peak_mb_abs":  round(peak_mb_abs, 1),
        "overhead_mb":  round(overhead_mb, 1),
        "mb_per_sample": round(mb_per_sample, 1),
        "kept_tokens":  round(kept, 1),
        "total_tokens": total,
    }


def print_report(results, batch_sizes):
    print("\n" + "=" * 88)
    print("  VRAM DIAGNOSTIC REPORT")
    print("=" * 88)

    for bs in batch_sizes:
        subset = [r for r in results if r["batch_size"] == bs]
        if not subset:
            continue
        print(f"\n  Fixed batch_size = {bs}")
        print(f"  {'kr':>5}  {'Kept':>5}  {'Base(MB)':>9}  {'Peak(MB)':>9}  "
              f"{'Ovhd(MB)':>9}  {'MB/samp':>8}  {'Calib':>7}  {'Delta':>7}")
        print("  " + "-" * 68)
        for r in sorted(subset, key=lambda x: -x["keep_ratio"]):
            kr    = r["keep_ratio"]
            m     = r["metrics"]
            calib = _CALIBRATED_MB_PER_SAMPLE.get(kr, "?")
            delta = round(m["mb_per_sample"] - calib, 1) if isinstance(calib, (int, float)) else "?"
            print(f"  {kr:>5.1f}  {m['kept_tokens']:>5.0f}  "
                  f"{m['baseline_mb']:>9.0f}  {m['peak_mb_abs']:>9.0f}  "
                  f"{m['overhead_mb']:>9.0f}  {m['mb_per_sample']:>8.1f}  "
                  f"{str(calib):>7}  {str(delta):>7}")

    print(f"\n  ANOMALY CHECK -- peak_mb should decrease as kr decreases")
    bs_ref = batch_sizes[-1]  # largest batch size
    subset = sorted([r for r in results if r["batch_size"] == bs_ref], key=lambda x: -x["keep_ratio"])
    prev_peak = None
    for r in subset:
        kr   = r["keep_ratio"]
        peak = r["metrics"]["peak_mb_abs"]
        flag = ""
        if prev_peak is not None and peak > prev_peak:
            flag = "  <<< ANOMALY"
        print(f"    kr={kr:.1f}  peak={peak:.0f} MB{flag}")
        prev_peak = peak

    # Suggested updated calibration (bs=16 measurement with 20% headroom)
    bs_meas = batch_sizes[-1]
    print(f"\n  SUGGESTED CALIBRATION UPDATE (from bs={bs_meas}, +20% headroom):")
    print(f"  _MB_PER_SAMPLE_BY_KR = {{")
    for r in sorted([r for r in results if r["batch_size"] == bs_meas],
                    key=lambda x: -x["keep_ratio"]):
        kr  = r["keep_ratio"]
        mps = r["metrics"]["mb_per_sample"]
        sug = int(math.ceil(mps * 1.2 / 50) * 50)
        print(f"    {kr}: {sug},   # measured={mps:.0f} MB/sample")
    print("  }")
    print("=" * 88)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kr_list",     type=float, nargs="+", default=[1.0, 0.7, 0.4, 0.1])
    p.add_argument("--batch_sizes", type=int,   nargs="+", default=[4, 16])
    p.add_argument("--checkpoint",  default=CKPT_DIR)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    print(f"\n  Device: {device}  bf16={use_bf16}")
    if torch.cuda.is_available():
        props  = torch.cuda.get_device_properties(0)
        tot_gb = props.total_memory / 1024 ** 3
        print(f"  GPU: {props.name}  Total VRAM: {tot_gb:.1f} GB")

    print(f"\n  Building model ...")
    model = build_and_load_model(
        checkpoint_dir=args.checkpoint,
        keep_ratio=args.kr_list[0],
        use_merging=False,
        projector_hidden_dim=2048,
        ia_hidden_dim=512,
        llava15_model_name="llava-hf/llava-1.5-7b-hf",
        device=device,
        use_bf16=use_bf16,
    )

    base_alloc = mb(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
    print(f"  Model loaded -- allocated: {base_alloc:.0f} MB")

    clip_tokenizer = model.text_encoder.tokenizer if hasattr(model.text_encoder, "tokenizer") else None
    if clip_tokenizer is None:
        from transformers import CLIPTokenizer
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")

    llm_tokenizer = model.llm.tokenizer
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    results = []

    for kr in args.kr_list:
        model.set_keep_ratio(kr)
        for bs in args.batch_sizes:
            print(f"\n  Measuring kr={kr:.2f}  bs={bs} ...", flush=True)
            try:
                m = measure_vram(model, bs, device, clip_tokenizer, llm_tokenizer)
                print(f"    tokens={m['kept_tokens']:.0f}/{m['total_tokens']}  "
                      f"peak={m['peak_mb_abs']:.0f} MB  "
                      f"overhead={m['overhead_mb']:.0f} MB  "
                      f"MB/sample={m['mb_per_sample']:.1f}  "
                      f"(calib={_CALIBRATED_MB_PER_SAMPLE.get(kr, '?')})")
                results.append({"keep_ratio": kr, "batch_size": bs, "metrics": m})
            except RuntimeError as e:
                print(f"    ERROR: {e}")
                results.append({"keep_ratio": kr, "batch_size": bs, "metrics": {
                    "error": str(e), "baseline_mb": 0, "peak_mb_abs": 0,
                    "overhead_mb": 0, "mb_per_sample": 0, "kept_tokens": 0, "total_tokens": 0,
                }})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print_report(results, args.batch_sizes)

    out_path = os.path.join(BASE_DIR, "eval_outputs_paper", "vram_diagnostic.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Raw results saved -> {out_path}")


if __name__ == "__main__":
    main()
