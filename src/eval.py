import os
import re
import json
import math
import time
import random
import argparse
import collections
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import CLIPTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from models.pruning_vlm import PruningVLM
from utils.misc import set_seed


# ============================================================
# Answer normalization / metrics
# ============================================================
CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
    "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
    "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
    "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all",
    "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
    "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
    "youre": "you're", "youve": "you've"
}

MANUAL_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

ARTICLES = {"a", "an", "the"}
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
PUNCT = [
    ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\",
    "_", "-", ">", "<", "@", "`", ",", "?", "!"
]


def normalize_vqa_answer(s: str) -> str:
    s = s.lower().replace("\n", " ").replace("\t", " ").strip()
    s = COMMA_STRIP.sub(r"\1\3", s)
    s = PERIOD_STRIP.sub("", s)

    for p in PUNCT:
        if (p + " " in s) or (" " + p in s) or (re.search(COMMA_STRIP, s) is not None):
            s = s.replace(p, "")
        else:
            s = s.replace(p, " ")

    words = []
    for word in s.split():
        word = MANUAL_MAP.get(word, word)
        word = CONTRACTIONS.get(word, word)
        if word not in ARTICLES:
            words.append(word)

    s = " ".join(words)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_exact_match(pred: str, gt: str) -> float:
    return float(pred.strip() == gt.strip())


def compute_normalized_exact_match(pred: str, gt: str) -> float:
    return float(normalize_vqa_answer(pred) == normalize_vqa_answer(gt))


def compute_token_f1(pred: str, gt: str) -> float:
    pred_tokens = normalize_vqa_answer(pred).split()
    gt_tokens = normalize_vqa_answer(gt).split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    pred_counter = collections.Counter(pred_tokens)
    gt_counter = collections.Counter(gt_tokens)
    common = pred_counter & gt_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / max(1, len(pred_tokens))
    recall = num_same / max(1, len(gt_tokens))
    return 2 * precision * recall / max(1e-8, precision + recall)


def compute_vqa_soft_accuracy(pred: str, gt_answers: List[str]) -> float:
    pred_norm = normalize_vqa_answer(pred)
    gt_norm = [normalize_vqa_answer(x) for x in gt_answers]
    count = sum(1 for x in gt_norm if x == pred_norm)
    return min(1.0, count / 3.0)


# ============================================================
# Utils
# ============================================================
def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def attach_lora_to_llm(
    llm_wrapper,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    if not hasattr(llm_wrapper, "model"):
        raise ValueError("Expected model.llm to have attribute `.model`.")

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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    peft_llm = get_peft_model(base_llm, lora_config)
    llm_wrapper.model = peft_llm
    return llm_wrapper


# ============================================================
# Prompt / dataset helpers
# ============================================================
_VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def build_vqa_prompt(question: str) -> str:
    return f"{_VICUNA_SYSTEM} USER: <image>\n{question}\nASSISTANT:"


class VQAv2EvalDataset(Dataset):
    def __init__(
        self,
        questions_path: str,
        image_root: str,
        annotations_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        shuffle_before_select: bool = False,
        seed: int = 42,
        require_image: bool = True,
    ):
        super().__init__()
        self.image_root = image_root
        self.require_image = require_image

        with open(questions_path, "r", encoding="utf-8") as f:
            q_data = json.load(f)
        questions = q_data["questions"]

        ann_map = {}
        if annotations_path is not None:
            with open(annotations_path, "r", encoding="utf-8") as f:
                a_data = json.load(f)

            for ann in a_data["annotations"]:
                qid = ann["question_id"]
                answers = [x["answer"] for x in ann.get("answers", [])]
                mc_answer = ann.get("multiple_choice_answer", "")
                ann_map[qid] = {
                    "answers": answers,
                    "multiple_choice_answer": mc_answer,
                    "answer_type": ann.get("answer_type", None),
                    "question_type": ann.get("question_type", None),
                }

        samples = []
        image_root_lower = os.path.basename(image_root).lower()

        for q in questions:
            qid = q["question_id"]
            image_id = q["image_id"]
            question = str(q["question"]).strip()

            if "test2015" in image_root_lower:
                image_name = f"COCO_test2015_{image_id:012d}.jpg"
            elif "val2014" in image_root_lower:
                image_name = f"COCO_val2014_{image_id:012d}.jpg"
            elif "train2014" in image_root_lower:
                image_name = f"COCO_train2014_{image_id:012d}.jpg"
            else:
                image_name = f"COCO_val2014_{image_id:012d}.jpg"

            image_path = os.path.join(image_root, image_name)
            if require_image and not os.path.exists(image_path):
                continue

            item = {
                "image_path": image_path,
                "question": question,
                "prompt_text": build_vqa_prompt(question),
                "question_id": qid,
                "image_id": image_id,
            }

            if qid in ann_map:
                item["answers"] = ann_map[qid]["answers"]
                item["multiple_choice_answer"] = str(ann_map[qid]["multiple_choice_answer"]).strip().lower()
                item["answer_type"] = ann_map[qid]["answer_type"]
                item["question_type"] = ann_map[qid]["question_type"]
            else:
                item["answers"] = None
                item["multiple_choice_answer"] = None
                item["answer_type"] = None
                item["question_type"] = None

            samples.append(item)

        if shuffle_before_select:
            rng = random.Random(seed)
            rng.shuffle(samples)

        if max_samples is not None:
            samples = samples[:max_samples]

        self.samples = samples
        print(f"[VQAv2EvalDataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        return {
            "image": image,
            "question": item["question"],
            "prompt_text": item["prompt_text"],
            "question_id": item["question_id"],
            "image_id": item["image_id"],
            "answers": item["answers"],
            "multiple_choice_answer": item["multiple_choice_answer"],
            "answer_type": item["answer_type"],
            "question_type": item["question_type"],
        }


def vqa_eval_collate_fn(batch: List[Dict[str, Any]]):
    return {
        "image": [x["image"] for x in batch],
        "question": [x["question"] for x in batch],
        "prompt_text": [x["prompt_text"] for x in batch],
        "question_id": [x["question_id"] for x in batch],
        "image_id": [x["image_id"] for x in batch],
        "answers": [x["answers"] for x in batch],
        "multiple_choice_answer": [x["multiple_choice_answer"] for x in batch],
        "answer_type": [x["answer_type"] for x in batch],
        "question_type": [x["question_type"] for x in batch],
    }


# ============================================================
# Input prep
# ============================================================
def build_image_transform(image_size=336):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def prepare_images_for_model(images, device, image_size=336):
    tfm = build_image_transform(image_size=image_size)
    image_tensors = [tfm(img) for img in images]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors.to(device, non_blocking=True)


def build_llm_batch_from_prompt_answer(
    prompt_texts: List[str],
    answers: List[str],
    tokenizer,
    max_length: int = 256,
):
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id must not be None.")

    if tokenizer.eos_token is None and tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have eos token or eos_token_id.")

    eos_text = tokenizer.eos_token if tokenizer.eos_token is not None else ""

    batch_input_ids = []
    batch_labels = []
    valid_indices = []

    num_truncated = 0
    num_empty_answer_after_trunc = 0

    for i, (prompt_text, answer) in enumerate(zip(prompt_texts, answers)):
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=True,   # BOS <s> required by Vicuna
            truncation=False,
        )["input_ids"]

        answer_text = " " + str(answer).strip().lower()
        if eos_text:
            answer_text = answer_text + eos_text

        answer_ids = tokenizer(
            answer_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        if len(answer_ids) == 0 and tokenizer.eos_token_id is not None:
            answer_ids = [tokenizer.eos_token_id]

        max_prompt_len = max_length - 1
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[:max_prompt_len]
            num_truncated += 1

        remaining = max_length - len(prompt_ids)
        truncated_answer_ids = answer_ids[:remaining]

        if len(truncated_answer_ids) < len(answer_ids):
            num_truncated += 1

        if len(truncated_answer_ids) == 0:
            num_empty_answer_after_trunc += 1
            continue

        input_ids = prompt_ids + truncated_answer_ids
        labels = ([-100] * len(prompt_ids)) + truncated_answer_ids

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        valid_indices.append(i)

    if len(batch_input_ids) == 0:
        return None

    pad_id = tokenizer.pad_token_id
    max_seq_len = max(len(x) for x in batch_input_ids)

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for input_ids, labels in zip(batch_input_ids, batch_labels):
        pad_len = max_seq_len - len(input_ids)
        padded_input_ids.append(input_ids + [pad_id] * pad_len)
        padded_attention_mask.append([1] * len(input_ids) + [0] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "num_truncated": num_truncated,
        "num_empty_answer_after_trunc": num_empty_answer_after_trunc,
        "num_valid_samples": len(batch_input_ids),
        "valid_indices": valid_indices,
    }


# ============================================================
# Model helpers
# ============================================================
def build_model(
    keep_ratio: float,
    use_bf16: bool,
    gate_temperature: float = 0.2,
    use_ste: bool = False,
):
    # Use LLaVA-1.5 as backbone — LLM + projector pretrained weights are loaded
    # automatically in LLaVALM.__init__ when model_name contains 'llava'.
    # llm_scale_visual_prefix=False: the LLaVA-1.5 projector already outputs
    # visual tokens at the correct scale for the frozen LLaVA-1.5 LLM.
    model = PruningVLM(
        clip_model_name="openai/clip-vit-large-patch14-336",
        llm_model_name="llava-hf/llava-1.5-7b-hf",
        keep_ratio=keep_ratio,
        alpha=0.6,
        learnable_alpha=True,
        llm_torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        gate_temperature=gate_temperature,
        gate_score_scale=5.0,
        use_ste=use_ste,
        gate_threshold_mode="topk",
        llm_use_grad=False,
        llm_scale_visual_prefix=False,
        llm_visual_prefix_scale=1.0,
    )

    freeze_module(model.vision_encoder)
    freeze_module(model.text_encoder)
    return model


def load_lightweight_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    device: str,
):
    training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
    non_llm_path = os.path.join(checkpoint_dir, "non_llm_trainables.pt")
    lora_dir = os.path.join(checkpoint_dir, "llm_lora_adapter")

    metadata = None
    if os.path.exists(training_state_path):
        metadata = torch.load(training_state_path, map_location=device)
        print(f"Loaded training metadata from: {training_state_path}")
        # Strip large/non-serializable entries (optimizer/scheduler state dicts) before printing
        _printable = {
            k: v for k, v in metadata.items()
            if k not in ("optimizer_state_dict", "scheduler_state_dict")
            and not isinstance(v, torch.Tensor)
        }
        print(json.dumps(_printable, indent=2, ensure_ascii=False))
    else:
        print(f"[Warning] training_state.pt not found at: {training_state_path}")

    if os.path.exists(non_llm_path):
        non_llm_ckpt = torch.load(non_llm_path, map_location=device)
        ckpt_state = non_llm_ckpt.get("trainable_non_llm_state_dict", {})
        model_state = model.state_dict()

        compatible = {}
        skipped = []
        for k, v in ckpt_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                compatible[k] = v
            else:
                skipped.append(
                    f"{k}: ckpt={tuple(v.shape) if hasattr(v, 'shape') else '?'} "
                    f"model={tuple(model_state[k].shape) if k in model_state else 'missing'}"
                )

        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(f"Loaded non-LLM trainables from: {non_llm_path}")
        print(f"Compatible keys loaded: {len(compatible)}")
        print(f"Missing keys         : {len(missing)}")
        print(f"Unexpected keys      : {len(unexpected)}")
        if skipped:
            print(f"Skipped mismatch/missing keys: {len(skipped)}")
            preview = skipped[:20]
            for item in preview:
                print(f"  - {item}")
            if len(skipped) > len(preview):
                print(f"  ... and {len(skipped) - len(preview)} more")
    else:
        print(f"[Warning] Not found: {non_llm_path}")
        print("[Info] Continue with pure pretrained backbone (no scorer/projector checkpoint loaded).")

    if os.path.exists(lora_dir):
        model.llm.model.load_adapter(lora_dir, adapter_name="default")
        print(f"Loaded LoRA adapter from: {lora_dir}")
    else:
        # freeze_llm=True training path: no LoRA adapter saved.
        # LLM stays as pretrained LLaVA-1.5 weights — no action needed.
        print(f"[Info] No LoRA adapter at {lora_dir} — using pretrained LLM weights.")

    return model, metadata


# ============================================================
# Generation helper
# ============================================================
def clean_generated_text(text: str) -> str:
    original = text
    text = text.strip()

    # lấy phần sau "Answer:" nếu có
    m = re.search(r"answer\s*[:\-]\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m is not None:
        text = m.group(1).strip()

    text = text.split("\n")[0].strip()
    text = re.sub(r"\s+", " ", text).strip()

    # Strip leading tokenization artifacts: commas, apostrophes, quotes, semicolons
    # that appear before the real answer (e.g. ", no" → "no", "' yes" → "yes")
    text = re.sub(r"^[\s,;'\"\.:]+", "", text).strip()
    # Strip trailing artifacts: quotes, apostrophes
    text = re.sub(r"[\s'\"]+$", "", text).strip()

    # Fallback: if aggressive cleanup removes everything, keep first non-empty line.
    if len(text) == 0:
        for line in original.splitlines():
            line = line.strip()
            if line:
                return re.sub(r"^[\s,;'\"\.:]+", "", re.sub(r"\s+", " ", line)).strip()
    return text


def decode_new_tokens_only(
    tokenizer,
    generated_ids: torch.Tensor,
    max_new_tokens: int,
) -> List[str]:
    """
    Decode only newly generated tail tokens.

    With inputs_embeds-based generation, decoded full sequences can include
    prompt/prefix artifacts that later get cleaned into empty strings.
    Decoding only the tail stabilizes answer extraction.
    """
    if generated_ids.ndim != 2:
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    tail_len = max(1, int(max_new_tokens))
    if generated_ids.size(1) > tail_len:
        tail_ids = generated_ids[:, -tail_len:]
    else:
        tail_ids = generated_ids

    texts = tokenizer.batch_decode(tail_ids, skip_special_tokens=True)
    return texts


@torch.no_grad()
def generate_answers(
    model: PruningVLM,
    images: torch.Tensor,
    clip_input_ids: torch.Tensor,
    clip_attention_mask: torch.Tensor,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    max_new_tokens: int = 16,
    num_beams: int = 3,
    use_hard_pruning: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    cls_emb, visual_tokens = model.vision_encoder(images)
    text_tokens = model.text_encoder(clip_input_ids, clip_attention_mask)

    cls_score_tokens = model.vision_encoder.project_features(cls_emb)
    visual_score_tokens = model.vision_encoder.project_features(visual_tokens)
    z = model.projector(visual_tokens)

    # ── Full scoring pipeline — must exactly match pruning_vlm.forward() ──
    # 1) Text importance weights β
    beta = model.text_importance(text_tokens, clip_attention_mask)   # [B, L]

    # 2) CLS-style scorer in frozen CLIP joint space
    s_cls = model.cls_scorer(
        cls_embedding=cls_score_tokens,
        patch_tokens=visual_score_tokens,
    )  # [B, N]

    # 3) Instruction-aware scorer (uses β for weighted pooling)
    s_ia, A = model.instruction_aware(
        text_tokens=text_tokens,
        visual_tokens=visual_score_tokens,
        beta=beta,
        text_attention_mask=clip_attention_mask,
    )  # [B, N]

    # 4) Score fusion: S = alpha * norm(S_ia) + (1-alpha) * norm(S_cls)
    text_cls = text_tokens[:, 0, :] if getattr(model, "question_conditioned_alpha", False) else None
    fused_scores, _ = model.score_fusion(s_ia, s_cls, text_cls_token=text_cls)  # [B, N], [B,1] or scalar

    dynamic_keep_ratio = model.get_dynamic_keep_ratio(
        z,
        text_tokens=text_tokens,
        text_attention_mask=clip_attention_mask,
    )
    soft_gates, tau = model._build_soft_gates(fused_scores, keep_ratios=dynamic_keep_ratio)
    hard_mask, topk_idx, topk_scores = model._build_hard_mask(
        fused_scores,
        keep_ratios=dynamic_keep_ratio,
    )

    if use_hard_pruning:
        visual_for_llm, kept_indices, kept_scores = model.token_pruner(z, fused_scores)
        if dynamic_keep_ratio is not None:
            visual_for_llm, kept_indices, kept_scores = model.token_pruner(
                z,
                fused_scores,
                keep_ratio=dynamic_keep_ratio,
            )
        gate_mask_used = hard_mask
    else:
        gate_mask_used = soft_gates
        visual_for_llm = z * gate_mask_used.unsqueeze(-1)
        kept_indices = topk_idx
        kept_scores = topk_scores

    llm_model = model.llm.model
    input_embed_layer = llm_model.get_input_embeddings()

    # Cast to LLM dtype — no magnitude rescaling: LLaVA-1.5 projector already
    # outputs tokens at the correct scale for the frozen LLaVA-1.5 LLM.
    visual_for_llm = visual_for_llm.to(
        device=next(input_embed_layer.parameters()).device,
        dtype=next(input_embed_layer.parameters()).dtype,
    )

    # ── Insert visual tokens INLINE at <image> token position (per-sample) ───
    # Per-sample processing handles variable padding (left or right) correctly.
    img_tok_id = getattr(model.llm, 'image_token_id', 32000)
    B = prompt_input_ids.size(0)
    K_vis = visual_for_llm.size(1)

    vis_mask_base = kept_indices.ge(0).to(dtype=prompt_attention_mask.dtype,
                                          device=prompt_attention_mask.device)  # [B, K]

    embed_list, mask_list = [], []
    for b in range(B):
        ids_b = prompt_input_ids[b]  # [L]
        positions = (ids_b == img_tok_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            p = positions[0].item()
            pre_e  = input_embed_layer(ids_b[:p].unsqueeze(0))    # [1, p, D]
            post_e = input_embed_layer(ids_b[p+1:].unsqueeze(0))  # [1, L-p-1, D]
            emb_b  = torch.cat([pre_e, visual_for_llm[b:b+1], post_e], dim=1)
            msk_b  = torch.cat([
                prompt_attention_mask[b, :p],
                vis_mask_base[b],
                prompt_attention_mask[b, p+1:],
            ], dim=0).unsqueeze(0)
        else:
            # Fallback: prepend
            text_e = input_embed_layer(ids_b.unsqueeze(0))
            emb_b  = torch.cat([visual_for_llm[b:b+1], text_e], dim=1)
            msk_b  = torch.cat([vis_mask_base[b], prompt_attention_mask[b]], dim=0).unsqueeze(0)
        embed_list.append(emb_b)
        mask_list.append(msk_b)

    inputs_embeds       = torch.cat(embed_list, dim=0)  # [B, L-1+K, D]
    full_attention_mask = torch.cat(mask_list,  dim=0)  # [B, L-1+K]

    generated_ids = llm_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=full_attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,
        num_beams=num_beams,
        do_sample=False,
        early_stopping=True,
        length_penalty=0.8,
        pad_token_id=model.llm.tokenizer.pad_token_id,
        eos_token_id=model.llm.tokenizer.eos_token_id,
    )

    texts = decode_new_tokens_only(
        tokenizer=model.llm.tokenizer,
        generated_ids=generated_ids,
        max_new_tokens=max_new_tokens,
    )
    texts = [clean_generated_text(x) for x in texts]

    info = {
        "fused_scores": fused_scores,
        "soft_gates": soft_gates,
        "hard_mask": hard_mask,
        "gate_mask_used": gate_mask_used,
        "projected_tokens": z,
        "visual_for_llm": visual_for_llm,
        "kept_indices": kept_indices,
        "kept_scores": kept_scores,
        "tau": tau,
        "A": A,
        "beta": beta,
        "s_cls": s_cls,
        "s_ia": s_ia,
        "dynamic_keep_ratio": dynamic_keep_ratio,
    }
    return texts, info


# ============================================================
# Core evaluation
# ============================================================
@torch.no_grad()
def evaluate(
    checkpoint_dir: str,
    questions_path: str,
    annotations_path: Optional[str],
    image_root: str,
    output_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    keep_ratio: Optional[float] = None,
    gate_temperature: float = 0.2,
    use_ste: bool = False,
    max_new_tokens: int = 16,
    num_beams: int = 3,
    image_size: int = 336,
    seed: int = 42,
    llm_max_length: int = 256,
    clip_max_length: int = 64,
):
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Build with a safe placeholder keep ratio; real eval ratio is resolved after loading metadata.
    init_keep_ratio = 0.5 if keep_ratio is None else keep_ratio
    model = build_model(
        keep_ratio=init_keep_ratio,
        use_bf16=use_bf16,
        gate_temperature=gate_temperature,
        use_ste=use_ste,
    )
    model, train_metadata = load_lightweight_checkpoint(model, checkpoint_dir=checkpoint_dir, device=device)
    model = model.to(device)
    model.eval()

    if isinstance(train_metadata, dict):
        dyn_enabled = bool(train_metadata.get("dynamic_budget_enabled", False))
        model.set_dynamic_budget(dyn_enabled)
        min_r = float(train_metadata.get("dynamic_budget_min_keep_ratio", 0.35))
        max_r = float(train_metadata.get("dynamic_budget_max_keep_ratio", 0.85))
        model.set_dynamic_budget_range(min_r, max_r)

    resolved_keep_ratio = keep_ratio
    if resolved_keep_ratio is None:
        if isinstance(train_metadata, dict) and ("keep_ratio" in train_metadata):
            resolved_keep_ratio = float(train_metadata["keep_ratio"])
        else:
            resolved_keep_ratio = float(getattr(model, "keep_ratio", 0.5))
    model.set_keep_ratio(float(resolved_keep_ratio))
    model.set_gate_temperature(gate_temperature)

    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")

    llm_tokenizer = model.llm.tokenizer
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    dataset = VQAv2EvalDataset(
        questions_path=questions_path,
        annotations_path=annotations_path,
        image_root=image_root,
        max_samples=max_samples,
        shuffle_before_select=False,
        seed=seed,
        require_image=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=vqa_eval_collate_fn,
        drop_last=False,
    )

    total_loss = 0.0
    total_loss_batches = 0

    total_correct_tokens = 0
    total_valid_tokens = 0

    total_exact_match = 0.0
    total_norm_exact_match = 0.0
    total_vqa_soft_acc = 0.0
    total_token_f1 = 0.0
    total_metric_samples = 0
    total_samples = 0

    total_projected_tokens = 0.0
    total_kept_tokens = 0.0
    total_soft_keep_ratio = 0.0
    total_hard_keep_ratio = 0.0
    total_attention_cost_ratio = 0.0

    total_truncated_samples = 0
    total_empty_answer_after_trunc = 0
    skipped_loss_batches = 0

    batch_times = []

    answer_type_stats = collections.defaultdict(lambda: {"count": 0, "vqa_acc": 0.0})
    question_type_stats = collections.defaultdict(lambda: {"count": 0, "vqa_acc": 0.0})

    records = []
    generated_examples = []
    empty_prediction_count = 0

    for batch_idx, batch in enumerate(dataloader):
        start_t = time.perf_counter()

        images = prepare_images_for_model(batch["image"], device=device, image_size=image_size)
        questions = batch["question"]
        prompt_texts = batch["prompt_text"]
        gt_answers_list = batch["answers"]
        gt_mc_answers = batch["multiple_choice_answer"]
        qids = batch["question_id"]
        img_ids = batch["image_id"]
        answer_types = batch["answer_type"]
        question_types = batch["question_type"]

        clip_tokens = clip_tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=clip_max_length,
        )
        clip_input_ids = clip_tokens.input_ids.to(device, non_blocking=True)
        clip_attention_mask = clip_tokens.attention_mask.to(device, non_blocking=True)

        prompt_tokens = llm_tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=llm_max_length,
        )
        prompt_input_ids = prompt_tokens.input_ids.to(device, non_blocking=True)
        prompt_attention_mask = prompt_tokens.attention_mask.to(device, non_blocking=True)

        has_gt = all(x is not None for x in gt_mc_answers)

        if has_gt:
            llm_batch = build_llm_batch_from_prompt_answer(
                prompt_texts=prompt_texts,
                answers=gt_mc_answers,
                tokenizer=llm_tokenizer,
                max_length=llm_max_length,
            )

            if llm_batch is not None:
                total_truncated_samples += llm_batch["num_truncated"]
                total_empty_answer_after_trunc += llm_batch["num_empty_answer_after_trunc"]

                valid_indices = llm_batch["valid_indices"]

                if len(valid_indices) != len(prompt_texts):
                    images_loss = images[valid_indices]
                    clip_input_ids_loss = clip_input_ids[valid_indices]
                    clip_attention_mask_loss = clip_attention_mask[valid_indices]
                else:
                    images_loss = images
                    clip_input_ids_loss = clip_input_ids
                    clip_attention_mask_loss = clip_attention_mask

                llm_input_ids = llm_batch["input_ids"].to(device, non_blocking=True)
                llm_attention_mask = llm_batch["attention_mask"].to(device, non_blocking=True)
                labels = llm_batch["labels"].to(device, non_blocking=True)

                valid_target_count = (labels != -100).sum().item()

                if valid_target_count > 0:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16 if use_bf16 else torch.float16,
                        enabled=torch.cuda.is_available(),
                    ):
                        outputs = model(
                            images=images_loss,
                            input_ids=clip_input_ids_loss,
                            attention_mask=clip_attention_mask_loss,
                            llm_input_ids=llm_input_ids,
                            llm_attention_mask=llm_attention_mask,
                            labels=labels,
                            return_intermediates=True,
                            use_hard_pruning=False,  # khớp train loss path
                        )

                    logits = outputs.get("logits", None)
                    loss = outputs.get("loss", None)

                    if loss is not None and torch.isfinite(loss):
                        total_loss += loss.item()
                        total_loss_batches += 1

                    if logits is not None and "visual_for_llm" in outputs:
                        num_visual_tokens = outputs["visual_for_llm"].size(1)
                        text_logits = logits[:, num_visual_tokens:, :].contiguous()

                        if text_logits.size(1) == labels.size(1):
                            shift_logits = text_logits[:, :-1, :].contiguous()
                            shift_labels = labels[:, 1:].contiguous()

                            pred_ids = shift_logits.argmax(dim=-1)
                            valid_mask = shift_labels.ne(-100)

                            correct = ((pred_ids == shift_labels) & valid_mask).sum().item()
                            valid = valid_mask.sum().item()

                            total_correct_tokens += correct
                            total_valid_tokens += valid
                else:
                    skipped_loss_batches += 1
            else:
                skipped_loss_batches += 1

        gen_texts, gen_info = generate_answers(
            model=model,
            images=images,
            clip_input_ids=clip_input_ids,
            clip_attention_mask=clip_attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            use_hard_pruning=True,
        )

        elapsed = time.perf_counter() - start_t
        batch_times.append(elapsed)

        projected_tokens = gen_info["projected_tokens"]
        hard_mask = gen_info["hard_mask"]
        soft_gates = gen_info["soft_gates"]

        B, N, _ = projected_tokens.shape
        kept_tokens_batch = hard_mask.sum(dim=-1).float()
        soft_keep_batch = soft_gates.mean(dim=-1)
        hard_keep_batch = hard_mask.mean(dim=-1)

        total_projected_tokens += float(B * N)
        total_kept_tokens += kept_tokens_batch.sum().item()
        total_soft_keep_ratio += soft_keep_batch.sum().item()
        total_hard_keep_ratio += hard_keep_batch.sum().item()
        total_attention_cost_ratio += ((kept_tokens_batch / N) ** 2).sum().item()

        for i in range(B):
            pred = clean_generated_text(gen_texts[i])
            if len(pred.strip()) == 0:
                empty_prediction_count += 1

            gt_answers = gt_answers_list[i] if gt_answers_list[i] is not None else None
            gt_mc = gt_mc_answers[i]

            record = {
                "question_id": int(qids[i]),
                "image_id": int(img_ids[i]),
                "question": questions[i],
                "prediction": pred,
                "ground_truth_multiple_choice": gt_mc,
                "ground_truth_answers": gt_answers,
                "answer_type": answer_types[i],
                "question_type": question_types[i],
                "num_projected_tokens": int(N),
                "num_kept_tokens": int(kept_tokens_batch[i].item()),
                "soft_keep_ratio": float(soft_keep_batch[i].item()),
                "hard_keep_ratio": float(hard_keep_batch[i].item()),
            }

            if gt_answers is not None and gt_mc is not None:
                em = compute_exact_match(pred, gt_mc)
                nem = compute_normalized_exact_match(pred, gt_mc)
                f1 = compute_token_f1(pred, gt_mc)
                vqa_acc = compute_vqa_soft_accuracy(pred, gt_answers)

                total_exact_match += em
                total_norm_exact_match += nem
                total_token_f1 += f1
                total_vqa_soft_acc += vqa_acc
                total_metric_samples += 1

                if answer_types[i] is not None:
                    answer_type_stats[answer_types[i]]["count"] += 1
                    answer_type_stats[answer_types[i]]["vqa_acc"] += vqa_acc

                if question_types[i] is not None:
                    question_type_stats[question_types[i]]["count"] += 1
                    question_type_stats[question_types[i]]["vqa_acc"] += vqa_acc

                record["exact_match"] = em
                record["normalized_exact_match"] = nem
                record["token_f1"] = f1
                record["vqa_soft_accuracy"] = vqa_acc

            records.append(record)
            total_samples += 1

            if len(generated_examples) < 100:
                generated_examples.append({
                    "question_id": int(qids[i]),
                    "image_id": int(img_ids[i]),
                    "question": questions[i],
                    "prediction": pred,
                    "ground_truth_multiple_choice": gt_mc,
                    "ground_truth_answers": gt_answers,
                    "num_projected_tokens": int(N),
                    "num_kept_tokens": int(kept_tokens_batch[i].item()),
                    "kept_indices_top10": [int(x) for x in gen_info["kept_indices"][i][:10].tolist()],
                    "kept_scores_top10": [float(x) for x in gen_info["kept_scores"][i][:10].tolist()],
                })

        if batch_idx % 20 == 0:
            avg_keep_so_far = total_kept_tokens / max(1, total_samples)
            avg_vqa_so_far = (
                total_vqa_soft_acc / max(1, total_metric_samples)
                if total_metric_samples > 0 else None
            )
            print(
                f"[Eval] batch {batch_idx}/{len(dataloader)} | "
                f"samples={total_samples} | "
                f"avg_kept_tokens={avg_keep_so_far:.2f} | "
                f"vqa_acc={avg_vqa_so_far if avg_vqa_so_far is not None else 'N/A'}"
            )

    avg_loss = total_loss / max(1, total_loss_batches) if total_loss_batches > 0 else None
    perplexity = math.exp(avg_loss) if avg_loss is not None and avg_loss < 20 else None
    token_accuracy = total_correct_tokens / max(1, total_valid_tokens) if total_valid_tokens > 0 else None

    avg_exact_match = total_exact_match / max(1, total_metric_samples) if total_metric_samples > 0 else None
    avg_norm_exact_match = total_norm_exact_match / max(1, total_metric_samples) if total_metric_samples > 0 else None
    avg_token_f1 = total_token_f1 / max(1, total_metric_samples) if total_metric_samples > 0 else None
    avg_vqa_soft_acc = total_vqa_soft_acc / max(1, total_metric_samples) if total_metric_samples > 0 else None

    avg_projected_tokens = total_projected_tokens / max(1, total_samples)
    avg_kept_tokens = total_kept_tokens / max(1, total_samples)
    avg_soft_keep_ratio = total_soft_keep_ratio / max(1, total_samples)
    avg_keep_ratio = total_hard_keep_ratio / max(1, total_samples)
    avg_attention_cost_ratio = total_attention_cost_ratio / max(1, total_samples)
    estimated_attention_saving = 1.0 - avg_attention_cost_ratio

    avg_latency_sec_per_batch = sum(batch_times) / max(1, len(batch_times))
    avg_samples_per_sec = total_samples / max(1e-8, sum(batch_times))
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available()
        else 0.0
    )

    answer_type_summary = {}
    for k, v in answer_type_stats.items():
        answer_type_summary[k] = {
            "count": v["count"],
            "vqa_soft_accuracy": v["vqa_acc"] / max(1, v["count"]),
        }

    question_type_summary = {}
    for k, v in question_type_stats.items():
        question_type_summary[k] = {
            "count": v["count"],
            "vqa_soft_accuracy": v["vqa_acc"] / max(1, v["count"]),
        }

    alpha_value = None
    if hasattr(model.score_fusion, "alpha"):
        alpha_value = float(torch.clamp(model.score_fusion.alpha.detach(), 0.0, 1.0).item())

    summary = {
        "checkpoint_dir": checkpoint_dir,
        "questions_path": questions_path,
        "annotations_path": annotations_path,
        "image_root": image_root,
        "num_samples": total_samples,
        "num_metric_samples": total_metric_samples,
        "num_loss_batches": total_loss_batches,
        "skipped_loss_batches": skipped_loss_batches,
        "keep_ratio": float(resolved_keep_ratio),
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "token_accuracy": token_accuracy,
        "generation_exact_match": avg_exact_match,
        "generation_normalized_exact_match": avg_norm_exact_match,
        "generation_token_f1": avg_token_f1,
        "vqa_soft_accuracy": avg_vqa_soft_acc,
        "empty_prediction_count": empty_prediction_count,
        "empty_prediction_rate": empty_prediction_count / max(1, total_samples),
        "avg_projected_tokens": avg_projected_tokens,
        "avg_kept_tokens": avg_kept_tokens,
        "avg_soft_keep_ratio": avg_soft_keep_ratio,
        "avg_keep_ratio": avg_keep_ratio,
        "avg_attention_cost_ratio": avg_attention_cost_ratio,
        "estimated_attention_saving": estimated_attention_saving,
        "avg_latency_sec_per_batch": avg_latency_sec_per_batch,
        "avg_samples_per_sec": avg_samples_per_sec,
        "peak_memory_mb": peak_memory_mb,
        "learned_alpha": alpha_value,
        "total_truncated_samples": total_truncated_samples,
        "total_empty_answer_after_trunc": total_empty_answer_after_trunc,
        "train_metadata": {
            k: v for k, v in (train_metadata or {}).items()
            if k not in ("optimizer_state_dict", "scheduler_state_dict")
            and not isinstance(v, torch.Tensor)
        } if isinstance(train_metadata, dict) else None,
        "answer_type_breakdown": answer_type_summary,
        "question_type_breakdown": question_type_summary,
    }

    summary_path = os.path.join(output_dir, "eval_summary.json")
    records_path = os.path.join(output_dir, "eval_records.jsonl")
    examples_path = os.path.join(output_dir, "generated_examples.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(records_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(generated_examples, f, ensure_ascii=False, indent=2)

    print("\n===== EVAL SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved records to: {records_path}")
    print(f"Saved examples to: {examples_path}")

    return summary, records, generated_examples


# ============================================================
# Test prediction export
# ============================================================
@torch.no_grad()
def export_test_predictions(
    checkpoint_dir: str,
    questions_path: str,
    image_root: str,
    output_json: str,
    batch_size: int = 4,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    keep_ratio: Optional[float] = None,
    gate_temperature: float = 0.2,
    use_ste: bool = False,
    max_new_tokens: int = 16,
    num_beams: int = 3,
    image_size: int = 336,
    seed: int = 42,
    llm_max_length: int = 256,
    clip_max_length: int = 64,
):
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    init_keep_ratio = 0.5 if keep_ratio is None else keep_ratio
    model = build_model(
        keep_ratio=init_keep_ratio,
        use_bf16=use_bf16,
        gate_temperature=gate_temperature,
        use_ste=use_ste,
    )
    model, train_metadata = load_lightweight_checkpoint(model, checkpoint_dir=checkpoint_dir, device=device)
    model = model.to(device)
    model.eval()
    if isinstance(train_metadata, dict):
        dyn_enabled = bool(train_metadata.get("dynamic_budget_enabled", False))
        model.set_dynamic_budget(dyn_enabled)
        min_r = float(train_metadata.get("dynamic_budget_min_keep_ratio", 0.35))
        max_r = float(train_metadata.get("dynamic_budget_max_keep_ratio", 0.85))
        model.set_dynamic_budget_range(min_r, max_r)
    resolved_keep_ratio = keep_ratio
    if resolved_keep_ratio is None:
        if isinstance(train_metadata, dict) and ("keep_ratio" in train_metadata):
            resolved_keep_ratio = float(train_metadata["keep_ratio"])
        else:
            resolved_keep_ratio = float(getattr(model, "keep_ratio", 0.5))
    model.set_keep_ratio(float(resolved_keep_ratio))
    model.set_gate_temperature(gate_temperature)

    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")
    llm_tokenizer = model.llm.tokenizer

    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    dataset = VQAv2EvalDataset(
        questions_path=questions_path,
        annotations_path=None,
        image_root=image_root,
        max_samples=max_samples,
        shuffle_before_select=False,
        seed=seed,
        require_image=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=vqa_eval_collate_fn,
        drop_last=False,
    )

    predictions = []

    for batch_idx, batch in enumerate(dataloader):
        images = prepare_images_for_model(batch["image"], device=device, image_size=image_size)
        questions = batch["question"]
        prompt_texts = batch["prompt_text"]
        qids = batch["question_id"]

        clip_tokens = clip_tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=clip_max_length,
        )
        clip_input_ids = clip_tokens.input_ids.to(device, non_blocking=True)
        clip_attention_mask = clip_tokens.attention_mask.to(device, non_blocking=True)

        prompt_tokens = llm_tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=llm_max_length,
        )
        prompt_input_ids = prompt_tokens.input_ids.to(device, non_blocking=True)
        prompt_attention_mask = prompt_tokens.attention_mask.to(device, non_blocking=True)

        gen_texts, _ = generate_answers(
            model=model,
            images=images,
            clip_input_ids=clip_input_ids,
            clip_attention_mask=clip_attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            use_hard_pruning=True,
        )

        for qid, pred in zip(qids, gen_texts):
            pred = clean_generated_text(pred)
            predictions.append({
                "question_id": int(qid),
                "answer": pred,
            })

        if batch_idx % 20 == 0:
            print(f"[Test Export] batch {batch_idx}/{len(dataloader)} | predictions={len(predictions)}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Saved test predictions to: {output_json}")
    return predictions


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="val", choices=["val", "test"])

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/best_model")
    parser.add_argument("--questions_path", type=str, required=True)
    parser.add_argument("--annotations_path", type=str, default=None)
    parser.add_argument("--image_root", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="eval_outputs")
    parser.add_argument("--output_json", type=str, default="test_predictions.json")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--keep_ratio", type=float, default=None)
    parser.add_argument("--gate_temperature", type=float, default=0.2)
    parser.add_argument("--use_ste", action="store_true")

    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--llm_max_length", type=int, default=256)
    parser.add_argument("--clip_max_length", type=int, default=64)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "val":
        evaluate(
            checkpoint_dir=args.checkpoint_dir,
            questions_path=args.questions_path,
            annotations_path=args.annotations_path,
            image_root=args.image_root,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            keep_ratio=args.keep_ratio,
            gate_temperature=args.gate_temperature,
            use_ste=args.use_ste,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            image_size=args.image_size,
            seed=args.seed,
            llm_max_length=args.llm_max_length,
            clip_max_length=args.clip_max_length,
        )
    else:
        export_test_predictions(
            checkpoint_dir=args.checkpoint_dir,
            questions_path=args.questions_path,
            image_root=args.image_root,
            output_json=args.output_json,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            keep_ratio=args.keep_ratio,
            gate_temperature=args.gate_temperature,
            use_ste=args.use_ste,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            image_size=args.image_size,
            seed=args.seed,
            llm_max_length=args.llm_max_length,
            clip_max_length=args.clip_max_length,
        )


if __name__ == "__main__":
    main()
